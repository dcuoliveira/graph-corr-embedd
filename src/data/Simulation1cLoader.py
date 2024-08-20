import os
import sys
import pickle

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx

from tqdm import tqdm

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from utils.conn_data import load_pickle, load_pickle_fast

class Simulation1cLoader(object):
    """
    Class to load data from simulation 1. The description of the simulation 1
    is given at Fujita et al. (2017) paper.

    References
    ----------
    Fujita, A. et al. (2017). Correlation between graphs with an application to brain network analysis.

    """
    
    def __init__(self, graph_name: str, name: str="simulation1", sample: bool=False):
        super().__init__()
    
        self.graph_name = graph_name
        self.name = name
        self._read_data(sample=sample)

    def _read_data(self, sample: bool=False):
        if sample:
            self.graph_data = load_pickle_fast(os.path.join(os.path.dirname(__file__), "inputs", self.name, self.graph_name, "sample_graph_info.pkl"))
        else:
            self.graph_data = load_pickle_fast(os.path.join(os.path.dirname(__file__), "inputs", self.name, self.graph_name, "all_graph_info.pkl"))
                
    def save_processed_graph_data(self, graph_data_list):
        save_dir = os.path.join(os.path.dirname(__file__), "data", "inputs", "simulation1c", self.graph_name)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "all_graph_info_processed.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(graph_data_list, f)
        print(f"Processed graph data saved to {file_path}")

    def load_processed_graph_data(self):
        file_path = os.path.join(os.path.dirname(__file__), "data", "inputs", "simulation1c", self.graph_name, "all_graph_info_processed.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                graph_data_list = pickle.load(f)
            print(f"Processed graph data loaded from {file_path}")
            return graph_data_list
        else:
            raise Exception('No processed data found!')

    def create_graph_list(self, load_preprocessed=False):
        graph_data_list = []

        if not load_preprocessed:
            for i, (cov_tag, graph_list) in enumerate(self.graph_data.items()):
                cov_val = float(cov_tag)

                for n_sim, graph_pair_info in tqdm(enumerate(graph_list), total=len(graph_list), desc=f"Processing Graphs for Cov {cov_tag}"):
                    graph1 = graph_pair_info['graph1']
                    graph2 = graph_pair_info['graph2']

                    # Convert NetworkX graphs to adjacency matrices
                    adj1 = torch.tensor(nx.adjacency_matrix(graph1).toarray())
                    adj2 = torch.tensor(nx.adjacency_matrix(graph2).toarray())

                    # Use rows of adjacency matrices as features
                    x1 = adj1.type(torch.float32)
                    x2 = adj2.type(torch.float32)

                    # Concatenate the edge indices for both graphs
                    edge_index = torch.cat([from_networkx(graph1).edge_index, from_networkx(graph2).edge_index + graph1.number_of_nodes()], dim=1)

                    # concatenate x1 and x2 creating a new dimension
                    x = torch.stack([x1, x2], dim=0)

                    if cov_val != np.round(graph_pair_info["cov"], 1):
                        raise ValueError(f"Covariance value does not match: {cov_val}, {n_sim}")
                    
                    # Create a single Data object
                    data = Data(x=x,
                                edge_index=edge_index,
                                y=torch.round(torch.tensor([cov_val], dtype=torch.float), decimals=1),
                                n_simulations=graph_pair_info["n_simulations"],
                                n_graphs=graph_pair_info["n_graphs"])

                    graph_data_list.append(data)

            self.n_simulations = np.unique([data.n_simulations for data in graph_data_list])
            self.covs = np.sort(np.unique([np.round(data.y.item(), 1) for data in graph_data_list]))
            self.save_processed_graph_data(graph_data_list)
            return graph_data_list
        else:
            graph_data_list = self.load_processed_graph_data()
            self.n_simulations = np.unique([data.n_simulations for data in graph_data_list])
            self.covs = np.sort(np.unique([np.round(data.y.item(), 1) for data in graph_data_list]))
            return graph_data_list

    def create_graph_loader(self, batch_size: int=1):
        graph_data_list = []

        for i, (cov_tag, graph_list) in enumerate(self.graph_data.items()):
            cov_val = float(cov_tag)

            for n_sim, graph_pair_info in enumerate(graph_list):

                graph1 = graph_pair_info['graph1']
                graph2 = graph_pair_info['graph2']

                # Convert NetworkX graphs to adjacency matrices
                adj1 = torch.tensor(nx.adjacency_matrix(graph1).toarray())
                adj2 = torch.tensor(nx.adjacency_matrix(graph2).toarray())

                # Use rows of adjacency matrices as features
                x1 = adj1.type(torch.float32)
                x2 = adj2.type(torch.float32)

                # Concatenate the edge indices for both graphs
                edge_index = torch.cat([from_networkx(graph1).edge_index, from_networkx(graph2).edge_index + graph1.number_of_nodes()], dim=1)

                # concatenate x1 and x2 creating a new dimension
                x = torch.stack([x1, x2], dim=0)

                if cov_val != np.round(graph_pair_info["cov"], 1):
                    raise ValueError(f"Covariance value does not match: {cov_val}, {n_sim}")
                
                # Create a single Data object
                data = Data(x=x,
                            edge_index=edge_index,
                            y=torch.tensor([cov_val], dtype=torch.float),
                            n_simulations=graph_pair_info["n_simulations"],
                            n_graphs=graph_pair_info["n_graphs"])

                graph_data_list.append(data)

        # Create DataLoader
        loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=True)

        return loader

DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        import time

        start = time.time()

        loader = Simulation1Loader(sample=True)
        graph_loader = loader.create_graph_loader()

        # time to minutes
        print("Time to load and process data: ", (time.time() - start) / 60)
