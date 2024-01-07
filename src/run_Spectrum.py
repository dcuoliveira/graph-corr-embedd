import torch
import argparse
import os

from models.Spectrum import Spectrum
from data.Simulation1Loader import Simulation1Loader

from utils.conn_data import save_pickle

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--model_name', type=str, help='Model name.', default="spectrum")
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="simulation1")
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1)

if __name__ == '__main__':

    args = parser.parse_args()

    # define dataset
    sim = Simulation1Loader(name=args.dataset_name)
    loader = sim.create_graph_loader(batch_size=args.batch_size)

    # define model
    model = Spectrum()
    pred = []
    true = []
    for data in loader:

        # get inputs
        x1 = data.x[0, :, :]
        x2 = data.x[1, :, :]

        # forward pass
        embeddings1 = model.forward(x1)
        embeddings2 = model.forward(x2)

        # compute covariance between embeddings (true target)
        cov = model.compute_spearman_correlation(x=embeddings1, y=embeddings2)

        # store results
        pred.append(cov)
        true.append(data.y)
    
    pred = torch.stack(pred)
    true = torch.stack(true)
    results = {
        "pred": pred,
        "true": true
    }

    # check if file exists
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{args.model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save file
    save_pickle(path=f"{output_path}/pred.pkl", obj=results)
