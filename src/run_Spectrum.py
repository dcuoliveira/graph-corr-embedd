import torch
import argparse
import os

from models.Spectrum import Spectrum
from data.Simulation1Loader import Simulation1Loader

from utils.conn_data import save_pickle

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--model_name', type=str, help='Model name.', default="spectrum")
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=False)
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="simulation1")
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1)

if __name__ == '__main__':

    args = parser.parse_args()

    # define dataset
    sim = Simulation1Loader(name=args.dataset_name, sample=args.sample)
    loader = sim.create_graph_loader(batch_size=args.batch_size)

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
        inputs.append([x1, x2])
        embeddings.append([z1, z2])
    
    # pred list to tensor
    pred = torch.tensor(pred)
    true = torch.tensor(true)
    inputs = torch.tensor(inputs)
    embeddings = torch.tensor(embeddings)

    results = {
        "pred": pred,
        "true": true,
        "inputs": inputs,
        "embeddings": embeddings
    }

    # check if file exists
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{args.model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save file
    if args.sample:
        save_pickle(path=f"{output_path}/sample_results.pkl", obj=results)
    else:
        save_pickle(path=f"{output_path}/results.pkl", obj=results)
