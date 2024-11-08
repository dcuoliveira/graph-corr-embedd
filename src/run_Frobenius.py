import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from torch_geometric.data import DataLoader

from models.Frobenius import Frobenius
from data.Simulation1aLoader import Simulation1aLoader
from data.Simulation1cLoader import Simulation1cLoader

from utils.conn_data import save_pickle, save_inputs_piecewise
from utils.parsers import str_2_bool
from utils.activation_functions import scaled_arctan

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--model_name', type=str, help='Model name.', default="frobenius")
parser.add_argument('--graph_name', type=str, help='Graph name.', default="watts_strogatz")
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=True)
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="simulation1c")
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1)
parser.add_argument('--shuffle', type=str, help='Shuffle the dataset.', default=True)

if __name__ == '__main__':

    args = parser.parse_args()

    args.sample = str_2_bool(args.sample)

    # define dataset
    print('Loading the data from the simulation!')
    if args.dataset_name == "simulation1a":
        sim = Simulation1aLoader(name=args.dataset_name, sample=args.sample)
        dataset_list = sim.create_graph_list()

    elif args.dataset_name == "simulation1c":
        sim = Simulation1cLoader(name=args.dataset_name, sample=args.sample, graph_name = args.graph_name)
        print('Loading the simulation data!')
        dataset_list = sim.create_graph_list(load_preprocessed=True)
    else:
        raise Exception('Dataset not found!')
    print('Finish Loading')

    # define model
    model = Frobenius()
    pbar = tqdm(sim.n_simulations, total=len(sim.n_simulations), desc="Running Spectrum model")
    train_test_results = []
    for n in pbar:

        simulation_results = []
        for cov in sim.covs:

            filtered_data_list = [data for data in dataset_list if (data.n_simulations == n) and (data.y.item() == cov)]
            filtered_loader = DataLoader(filtered_data_list, batch_size=args.batch_size, shuffle=args.shuffle)

            sim_standardized_distances = []
            for data in filtered_loader:

                # get inputs
                x1 = data.x[0, :, :]
                x2 = data.x[1, :, :]

                # forward pass
                distance = model.forward(x1, x2)

                # standardized distance
                standardized_distance = scaled_arctan(distance)

                sim_standardized_distances.append(standardized_distance)
            sim_standardized_distances = torch.tensor(sim_standardized_distances)
            pred_cov = torch.mean(sim_standardized_distances)

            simulation_results.append([pred_cov.item(), cov])
        simulation_results = torch.tensor(simulation_results)

        train_test_results.append(simulation_results)
    train_test_results = torch.stack(train_test_results)

    results = {
        "train_test_results": train_test_results,
        "n_simulations": sim.n_simulations,
        "covs": sim.covs
    }

    # check if file exists
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{args.graph_name}/{args.model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # save file
    if args.sample:
        save_pickle(path=f"{output_path}/sample_results.pkl", obj=results)

    else:
        save_pickle(path=f"{output_path}/results.pkl", obj=results)
