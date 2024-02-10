import torch
import numpy as np
import argparse
import os

from models.Spectrum import Spectrum
from data.Simulation1aLoader import Simulation1aLoader

from utils.conn_data import save_pickle, save_inputs_piecewise
from utils.parsers import str_2_bool

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--model_name', type=str, help='Model name.', default="spectrum")
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=False)
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="simulation1a")
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1)

if __name__ == '__main__':

    args = parser.parse_args()

    args.sample = str_2_bool(args.sample)

    # define dataset
    sim = Simulation1aLoader(name=args.dataset_name, sample=args.sample)
    loaders = sim.create_graph_loader(batch_size=args.batch_size)
    for n_nodes, loader in loaders.items():
        
        # define model
        model = Spectrum()
        pred = []
        true = []
        inputs = []
        embeddings = []
        for data in loader:

            # get inputs
            x1 = data.x[0, :, :]
            x2 = data.x[1, :, :]

            # forward pass
            z1 = model.forward(x1)
            z2 = model.forward(x2)

            # compute covariance between embeddings (true target)
            cov = model.compute_spearman_correlation(x=z1, y=z2)

            # store results
            pred.append(cov)
            true.append(data.y)
            
            # check if cov is nan
            if np.isnan(cov):
                inputs.append([x1.numpy(), x2.numpy()])
                embeddings.append([z1, z2])
        
        # pred list to tensor
        pred = torch.tensor(pred)
        true = torch.tensor(true)

        if len(inputs) > 0:
            inputs = torch.tensor(inputs)
        else:
            inputs = None
        
        if len(embeddings) > 0:
            embeddings = torch.tensor(embeddings)
        else:
            embeddings = None

        inputs = {
            "inputs": inputs,
            "embeddings": embeddings,
        }

        results = {
            "pred": pred,
            "true": true,
        }

        # check if file exists
        output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{n_nodes}/{args.model_name}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # save file
        if args.sample:
            save_pickle(path=f"{output_path}/sample_inputs.pkl", obj=inputs)
            save_pickle(path=f"{output_path}/sample_results.pkl", obj=results)

        else:
            save_pickle(path=f"{output_path}/inputs.pkl", obj=inputs)
            save_pickle(path=f"{output_path}/results.pkl", obj=results)
