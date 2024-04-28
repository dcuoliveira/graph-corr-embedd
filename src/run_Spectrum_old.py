import torch
import numpy as np
import argparse
import os
import pandas as pd
from tqdm import tqdm

from models.Spectrum import Spectrum
from data.Simulation1Loader import Simulation1Loader

from utils.conn_data import save_pickle, save_inputs_piecewise
from utils.parsers import str_2_bool

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--model_name', type=str, help='Model name.', default="spectrum_old")
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=True)
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="simulation1")
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1)

if __name__ == '__main__':

    args = parser.parse_args()

    args.sample = str_2_bool(args.sample)

    # define dataset
    sim = Simulation1Loader(name=args.dataset_name, sample=args.sample)
    loader = sim.create_graph_loader(batch_size=args.batch_size)

    # define model
    model = Spectrum()
    outputs = []
    pbar = tqdm(loader, total=len(loader), desc="Running Spectrum model")
    for data in pbar:

        # get inputs
        x1 = data.x[0, :, :]
        x2 = data.x[1, :, :]

        # forward pass
        z1 = model.forward(x1)
        z2 = model.forward(x2)

        # save results
        outputs.append({"true": data.y.item(), "sr1": z1, "sr2": z2})
    outputs_df = pd.DataFrame(outputs)

    # compute covariance between embeddings (true target)
    train_test_results = []
    for true_cov in outputs_df["true"].unique():

        tmp_outputs_df = outputs_df.loc[outputs_df["true"] == true_cov]
        pred_cov = model.compute_spearman_rank_correlation(x=tmp_outputs_df['sr1'].values,
                                                           y=tmp_outputs_df['sr2'].values)

        train_test_results.append([pred_cov, true_cov])
    train_test_results = torch.tensor(train_test_results)

    results = {
        "train_test_results": train_test_results,
        "n_simulations": None,
        "covs": None
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
