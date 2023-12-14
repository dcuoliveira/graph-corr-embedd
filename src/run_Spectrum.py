import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import argparse
import os

from models.Spectrum import Spectrum
from data.ExamplesLoader import ExamplesLoader
from data.DataLoad import Dataload

from utils.conn_data import save_pickle

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--model_name', type=str, help='Model name.', default="spectrum")

if __name__ == '__main__':

    args = parser.parse_args()

    # load graph data
    el =  ExamplesLoader(example_name="cora")
    G, Adj, n_nodes = el.G, el.Adj, el.n_nodes

    # define model
    model = Spectrum()
    print( Adj.numpy().shape )
    print(model.compute_largest_eigenvalues( Adj.numpy() ) )

    

