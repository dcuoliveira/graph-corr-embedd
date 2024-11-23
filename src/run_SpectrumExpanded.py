import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from torch_geometric.data import DataLoader

from models.Spectrum import Spectrum
from data.Simulation1aLoader import Simulation1aLoader
from data.Simulation1cLoader import Simulation1cLoader

from utils.conn_data import save_pickle, save_inputs_piecewise
from utils.parsers import str_2_bool

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--model_name', type=str, help='Model name.', default="spectrumEx")
parser.add_argument('--graph_name', type=str, help='Graph name.', default="erdos_renyi")
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=False)
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="simulation1c")
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1)
parser.add_argument('--shuffle', type=str, help='Shuffle the dataset.', default=True)
parser.add_argument('--load_preprocessed', type=str, help='Load preprocessed graph data.', default=True)

if __name__ == '__main__':

    args = parser.parse_args()

    args.sample = str_2_bool(args.sample)
    args.load_preprocessed = str_2_bool(args.load_preprocessed)

    # define dataset
    if args.dataset_name == "simulation1a":
        sim = Simulation1aLoader(name=args.dataset_name, sample=args.sample)
        dataset_list = sim.create_graph_list()

    elif args.dataset_name == "simulation1c":
        sim = Simulation1cLoader(name=args.dataset_name, sample=args.sample, graph_name = args.graph_name)
        dataset_list = sim.create_graph_list(load_preprocessed=args.load_preprocessed)
    else:
        raise Exception('Dataset not found!')

    # define model
    model = Spectrum()
    pbar = tqdm(sim.n_simulations, total=len(sim.n_simulations), desc=f"Running {args.model_name} model")
    train_test_results = []
    for n in pbar:

        simulation_results = []
        for cov in sim.covs:

            filtered_data_list = [data for data in dataset_list if (data.n_simulations == n) and (data.y == cov)]
            filtered_loader = DataLoader(filtered_data_list, batch_size=args.batch_size, shuffle=args.shuffle)

            embeddings1 = []
            embeddings2 = [] 
            for data in filtered_loader:

                # get inputs
                x1 = data.x[0, :, :]
                x2 = data.x[1, :, :]

                # forward pass
                z1 = model.forward(x1, k=args.k)
                z2 = model.forward(x2, k=args.k)

                embeddings1 += list(z1)
                embeddings2 += list(z2)
            embeddings1 = torch.tensor(embeddings1)
            embeddings2 = torch.tensor(embeddings2)
            pred_cov = model.compute_spearman_rank_correlation(x=embeddings1, y=embeddings2)

            simulation_results.append([pred_cov, cov])
        simulation_results = torch.tensor(simulation_results)

        train_test_results.append(simulation_results)
    train_test_results = torch.stack(train_test_results)

    results = {
        "train_test_results": train_test_results,
        "n_simulations": sim.n_simulations,
        "covs": sim.covs
    }

    # check if file exists
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{args.graph_name}/{args.model_name}_k={args.k}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # save file
    if args.sample:
        save_pickle(path=f"{output_path}/sample_results.pkl", obj=results)

    else:
        save_pickle(path=f"{output_path}/results.pkl", obj=results)
