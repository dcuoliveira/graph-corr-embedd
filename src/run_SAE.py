import torch
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from torch_geometric.data import DataLoader

from models.SAE import StackedSparseAutoencoder
from loss_functions.LossReconSparse import LossReconSparse
from data.Simulation1Loader import Simulation1Loader
from utils.conn_data import save_pickle
from utils.parsers import str_2_bool

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--dataset_name', type=str, default="simulation1")
parser.add_argument('--sample', type=str, default=True)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--model_name', type=str, default="sae")
parser.add_argument('--input_size', type=int, default=100)
parser.add_argument('--hidden_sizes', type=list, default=[50,25,50])  # Comma-separated list for hidden layers
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--sparsity_penalty', type=float, default=1e-4)
parser.add_argument('--shuffle', type=str, help='Shuffle the dataset.', default=True)

if __name__ == '__main__':

    args = parser.parse_args()

    args.sample = str_2_bool(args.sample)

    # define dataset
    sim = Simulation1Loader(name=args.dataset_name, sample=args.sample)
    train_loader = sim.create_graph_loader(batch_size=args.batch_size)
    test_dataset_list = sim.create_graph_list()

    # define model
    model1 = StackedSparseAutoencoder(input_size=args.input_size,
                                      hidden_sizes=args.hidden_sizes,
                                      dropout=args.dropout,
                                      sparsity_penalty=args.sparsity_penalty)

    model2 = StackedSparseAutoencoder(input_size=args.input_size,
                                      hidden_sizes=args.hidden_sizes,
                                      dropout=args.dropout,
                                      sparsity_penalty=args.sparsity_penalty)
    
    # define optimizer
    optimizer1 = optim.Adam(model1.parameters(), lr=args.learning_rate)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.learning_rate)

    # define loss functions
    loss_func = LossReconSparse()

    # initialize tqdm
    pbar = tqdm(range(args.epochs))

    train_results = []
    xs_train, zs_train, z_norms_train = [], [], []
    epochs_loss_tot = []
    for epoch in pbar:

        epoch_loss1, epoch_loss2  = 0, 0
        epoch_results = []
        for data in train_loader:
            # get inputs
            x1 = data.x[0, :, :]
            x2 = data.x[1, :, :]

            # forward pass
            x1_hat, z1 = model1.forward(x1)
            x2_hat, z2 = model2.forward(x2)

            # compute correlation between embeddings (true target)
            pred_cov = model1.compute_spearman_rank_correlation(x=z1.flatten().detach(), y=z2.flatten().detach())

            # store pred and true values
            epoch_results.append([pred_cov, data.y])

            # compute loss
            loss1 = loss_func.forward(x1_hat, x1, model1)
            loss2 = loss_func.forward(x2_hat, x2, model2)

            # backward and optimize
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

        epoch_results = torch.tensor(epoch_results)

        # update tqdm
        pbar.update(1)
        pbar.set_description("Train Epoch: %d, Train Loss I & II: %.4f & %.4f" % (epoch, epoch_loss1, epoch_loss2))

        # save loss
        epochs_loss_tot.append([epoch_loss1, epoch_loss2])
        train_results.append(epoch_results)

    # pred list to tensor
    train_results = torch.stack(train_results)

    pbar = tqdm(sim.n_simulations, total=len(sim.n_simulations), desc="Running Spectrum model")
    test_results = []
    with torch.no_grad():
        for n in pbar:

            simulation_results = []
            for cov in sim.covs:

                filtered_data_list = [data for data in test_dataset_list if (data.n_simulations == n) and (data.y.item() == cov)]
                filtered_loader = DataLoader(filtered_data_list, batch_size=args.batch_size, shuffle=args.shuffle)

                embeddings = [] 
                for data in filtered_loader:
                    # get inputs
                    x1 = data.x[0, :, :]
                    x2 = data.x[1, :, :]

                    # forward pass
                    x1_hat, z1 = model1.forward(x1)
                    x2_hat, z2 = model2.forward(x2)
                    embeddings.append(torch.stack((z1.flatten().detach(), z2.flatten().detach()), dim=1))

                embeddings = torch.concat(embeddings)

                pred_cov = model1.compute_spearman_rank_correlation(x=embeddings[:,0], y=embeddings[:,1])

                simulation_results.append([pred_cov, cov])
            
            simulation_results = torch.tensor(simulation_results)
            test_results.append(simulation_results)
            
            pbar.update(1)
            pbar.set_description(f"Test Simulation: {n}")
            
    test_results = torch.stack(test_results)

    results = {
        "args": args,
        "train_results": train_results,
        "test_results": test_results,
        "train_loss": epochs_loss_tot,
    }

    model_name = f'{args.model_name}_{str(args.hidden_sizes)}_{int(args.epochs)}'

    # check if file exists
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save file
    if args.sample:
        save_pickle(path=f"{output_path}/sample_results.pkl", obj=results)
    else:
        save_pickle(path=f"{output_path}/results.pkl", obj=results)