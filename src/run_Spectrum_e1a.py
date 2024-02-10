import torch
import numpy as np
import argparse
import os
import pandas as pd

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
        outputs = []
        for data in loader:

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
        for true_cov in outputs_df["true"].unique():

            tmp_outputs_df = outputs_df.loc[outputs_df["true"] == true_cov]
            pred_cov = model.compute_spearman_rank_correlation(x=tmp_outputs_df['sr1'].values,
                                                               y=tmp_outputs_df['sr2'].values)

            outputs_df.loc[outputs_df["true"] == true_cov, "pred"] = pred_cov

        # store results
        pred = torch.tensor(outputs_df["pred"].values)
        true = torch.tensor(outputs_df["true"].values)

        inputs = {
            "inputs": outputs_df
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
