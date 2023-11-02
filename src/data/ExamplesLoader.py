import os
import numpy as np
import torch
import pandas as pd

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

    def _read_data(self):
        graph_data = load_pickle(os.path.join(os.path.dirname(__file__), "inputs", self.example_name, "graph_info.pickle"))

        self.G = graph_data["G"]
        self.Adj = graph_data["Adj"]
        self.n_nodes = graph_data["Node"]
    
DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        loader = ExamplesLoader()
        dataset = loader.get_dataset()
