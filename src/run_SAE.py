import torch
import torch.optim as optim
import argparse
import os
from tqdm import tqdm

from models.StackedSparseAutoencoder import StackedSparseAutoencoder
from data.Simulation1Loader import Simulation1Loader
from utils.conn_data import save_pickle
from utils.parsers import str_2_bool

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--dataset_name', type=str, default="simulation1")
parser.add_argument('--sample', type=str, default=False)
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
    model = StackedSparseAutoencoder(input_size=args.input_size,
                                     hidden_sizes=hidden_sizes,
                                     dropout=args.dropout,
                                     sparsity_penalty=args.sparsity_penalty)
    
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # initialize tqdm
    pbar = tqdm(range(args.epochs))

    for epoch in pbar:
        epoch_loss = 0
        for data in loader:
            x = data.x.view(-1, args.input_size)  # Adjust shape if necessary

            # forward pass
            recon_x, _ = model(x)

            # compute loss
            loss = model.loss_function(recon_x, x)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # update tqdm
        pbar.set_description(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss/len(loader):.4f}")

    # Save model and results
    model_name = f'{args.model_name}_{args.input_size}_{"_".join(map(str, hidden_sizes))}'
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Modify saving mechanism as per your requirement
    torch.save(model.state_dict(), f"{output_path}/model.pth")

    # Additional code for saving results, testing, etc. can be added here
