import os
import numpy as np
import torch
import pandas as pd
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data

from utils.conn_data import load_pickle

class ExamplesLoader(object):
    """
    This class implements the dataset used in Zhang, Zohren, and Roberts (2021)
    https://arxiv.org/abs/2005.13665 in to torch geomatric data loader format.
    The data consists of daily observatios of four etf prices and returns 
    concatenated together, from January 2000 to February 2023.
    
    """
    
    def __init__(self, example_name):
        super().__init__()
    
        self.example_name = example_name
        self._read_data()

        graph_data_loader = from_networkx(self.G)
        self.dataset = Data(x=self.graph_data, edge_index=graph_data_loader.edge_index)

    def _read_data(self):
        self.graph_data = load_pickle(os.path.join(os.path.dirname(__file__), "inputs", self.example_name, "graph_info.pickle"))

        self.G = self.graph_data["G"]
        self.Adj = self.graph_data["Adj"]
        self.n_nodes = self.graph_data["Node"]
        self.data = self.graph_data["torch_graph_data"]
    
DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        loader = ExamplesLoader()
        dataset = loader.get_dataset()
