import torch
import torch.optim as optim
import argparse
import os
from tqdm import tqdm

from models.SAE import StackedSparseAutoencoder
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
parser.add_argument('--hidden_sizes', type=str, default="50,25,50")  # Comma-separated list for hidden layers
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--sparsity_penalty', type=float, default=1e-4)

if __name__ == '__main__':

    args = parser.parse_args()
    args.sample = str_2_bool(args.sample)
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]

    # define dataset
    sim = Simulation1Loader(name=args.dataset_name, sample=args.sample)
    loader = sim.create_graph_loader(batch_size=args.batch_size)

    # define model
    model1    = StackedSparseAutoencoder(input_size=args.input_size,
                                        hidden_sizes=hidden_sizes,
                                        dropout=args.dropout,
                                        sparsity_penalty=args.sparsity_penalty)

    model2    = StackedSparseAutoencoder(input_size=args.input_size,
                                        hidden_sizes=hidden_sizes,
                                        dropout=args.dropout,
                                        sparsity_penalty=args.sparsity_penalty)



    # define optimizer
    optimizer1 = optim.Adam(model1.parameters(), lr=args.learning_rate)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.learning_rate)

    # initialize tqdm
    pbar = tqdm(range(args.epochs))
    train_pred, train_true = [], []
    xs_train, zs_train = [], []
    epoch_loss_train = []
    for epoch in pbar:
        epoch_loss1, epoch_loss2 = 0, 0 
        for data in loader:
            # get inputs
            x1 = data.x[0, :, :]
            x2 = data.x[1, :, :]

            # forward pass
            x1_hat, z1, = model1.forward(x1)
            x2_hat, z2  = model2.forward(x1)

            # compute correlation between embeddings (true target)
            corr = model1.compute_spearman_rank_correlation(x=z1.flatten().detach(), y=z2.flatten().detach())

            # compute loss
            loss1 = model1.loss_function(x1_hat, x1)
            loss2 = model2.loss_function(x2_hat, x2)

            # backward and optimize
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

        # update tqdm
        pbar.update(1)
        pbar.set_description("Train Epoch: %d, Train Loss I & II: %.4f & %.4f" % (epoch, epoch_loss1, epoch_loss2))

        # save loss
        epoch_loss_train.append([epoch_loss1, epoch_loss2])

    # pred list to tensor
    train_pred = torch.tensor(train_pred)
    train_true = torch.tensor(train_true)

    pbar = tqdm(enumerate(loader), total=len(loader))
    test_pred = []
    test_true = []
    with torch.no_grad():
        for s, data in pbar:
            # get inputs
            x1 = data.x[0, :, :]
            x2 = data.x[1, :, :]

            # forward pass
            x1_hat, z1 = model1.forward(x1)
            x2_hat, z2 = model2.forward(x2)

            # compute correlation between embeddings (true target)
            corr = model1.compute_spearman_rank_correlation(x=z1.flatten().detach(), y=z2.flatten().detach())

            # store pred and true values
            test_pred.append(corr)
            test_true.append(data.y)

            # update tqdm
            pbar.update(1)
            pbar.set_description(f"Test Sample: {s}")
        
    # pred list to tensor
    test_pred = torch.tensor(test_pred)
    test_true = torch.tensor(test_true)

    results = {
        "train_pred": train_pred,
        "train_true": train_true,
        "test_pred": test_pred,
        "test_true": test_true,
        "epoch_loss_train": epoch_loss_train,
    }

    model_name = f'{args.model_name}_{int(args.sparsity_penalty)}_{hidden_sizes}_{args.dropout}'
    # check if file exists
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save file
    if args.sample:
        save_pickle(path=f"{output_path}/sample_results.pkl", obj=results)
    else:
        save_pickle(path=f"{output_path}/results.pkl", obj=results)