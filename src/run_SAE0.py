import torch
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from torch_geometric.data import DataLoader

from models.SAE import StackedSparseAutoencoder
from loss_functions.LossReconSparse import LossReconSparse
from data.Simulation1aLoader import Simulation1aLoader
from utils.conn_data import save_pickle
from utils.parsers import str_2_bool, str_2_list

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--dataset_name', type=str, default="simulation1a")
parser.add_argument('--sample', type=str, default=False)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--model_name', type=str, default="sae0")
parser.add_argument('--input_size', type=int, default=100)
parser.add_argument('--hidden_sizes', type=str, default="50,25,50")  # Comma-separated list for hidden layers
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--sparsity_penalty', type=float, default=1e-4)
parser.add_argument('--shuffle', type=str, help='Shuffle the dataset.', default=True)

if __name__ == '__main__':

    args = parser.parse_args()

    args.sample = str_2_bool(args.sample)
    args.shuffle = str_2_bool(args.shuffle)
    args.hidden_sizes = str_2_list(args.hidden_sizes)

    # define dataset
    sim = Simulation1aLoader(name=args.dataset_name, sample=args.sample)
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

    epochs_predictions = []
    epochs_loss_tot = []
    for epoch in pbar:

        batch_loss_tot1, batch_loss_tot2 = [], []
        batch_predictions = []
        for data in train_loader:

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            # get inputs
            x1 = data.x[0, :, :]
            x2 = data.x[1, :, :]

            # forward pass
            x1_hat, z1 = model1.forward(x1)
            x2_hat, z2 = model2.forward(x2)

            # compute correlation between embeddings (true target)
            pred_cov = model1.compute_spearman_rank_correlation(x=z1.flatten().detach(), y=z2.flatten().detach())

            # store pred and true values
            batch_predictions.append([pred_cov, data.y])

            # compute loss
            loss1 = loss_func.forward(x1_hat, x1, model1)
            loss2 = loss_func.forward(x2_hat, x2, model2)

            # backward and optimize
            loss1.backward()
            optimizer1.step()

            loss2.backward()
            optimizer2.step()

            batch_loss_tot1.append(loss1.item())
            batch_loss_tot2.append(loss2.item())

        epochs_predictions.append(torch.tensor(batch_predictions))
        epochs_loss_tot.append(torch.stack([torch.tensor(batch_loss_tot1), torch.tensor(batch_loss_tot2)], axis=1))

        # update tqdm
        pbar.update(1)
        pbar.set_description("SAE Train Epoch: %d, Train Loss I & II: %.4f & %.4f" % (epoch, batch_loss_tot1[-1], batch_loss_tot2[-1]))

    epochs_predictions = torch.stack(epochs_predictions)
    epochs_loss_tot = torch.stack(epochs_loss_tot)

    pbar = tqdm(sim.n_simulations, total=len(sim.n_simulations), desc="Running SAE model on Test Data")
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
            
    test_predictions = torch.stack(test_results)

    args = {
        "args": args
    }

    predictions = {
        "train_predictions": epochs_predictions,
        "test_predictions": test_results,
    }

    training_info = {
        "train_loss": epochs_tot_loss,
        "epochs_global_loss": epochs_global_loss,
        "epochs_local_loss": epochs_local_loss,
        "epochs_reg_loss": epochs_reg_loss,
    }

    model_name = f'{args.model_name}_{int(args.n_hidden)}_{int(args.n_layers_enc)}_{int(args.n_layers_dec)}_{int(args.epochs)}'

    # check if file exists
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save file
    if args.sample:
        save_pickle(path=f"{output_path}/sample_args.pkl", obj=args)
        save_pickle(path=f"{output_path}/sample_predictions.pkl", obj=predictions)
        save_pickle(path=f"{output_path}/sample_training_info.pkl", obj=training_info)
        torch.save(model1.state_dict(), f"{output_path}/model1_sample.pth")
        torch.save(model2.state_dict(), f"{output_path}/model2_sample.pth")
    else:
        save_pickle(path=f"{output_path}/args.pkl", obj=args)
        save_pickle(path=f"{output_path}/predictions.pkl", obj=predictions)
        save_pickle(path=f"{output_path}/training_info.pkl", obj=training_info)
        torch.save(model1.state_dict(), f"{output_path}/model1.pth")
        torch.save(model2.state_dict(), f"{output_path}/model2.pth")