import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
import concurrent.futures
import numpy as np

from src.utils.conn_data import load_pickle

class Simulation1Loader(object):
    """
    Class to load data from simulation 1. The description of the simulation 1
    is given at Fujita et al. (2017) paper.

    References
    ----------
    Fujita, A. et al. (2017). Correlation between graphs with an application to brain network analysis.

    """
    
    def __init__(self, name: str="simulation1", sample: bool=False):
        super().__init__()
    
        self.name = name
        self._read_data(sample=sample)

    def _read_data(self, sample: bool=False):
        if sample:
            self.graph_data = load_pickle(os.path.join(os.path.dirname(__file__), "inputs", self.name, "sample_graph_info.pkl"))
        else:
            self.graph_data = load_pickle(os.path.join(os.path.dirname(__file__), "inputs", self.name, "all_graph_info.pkl"))

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
                data = Data(x=x, edge_index=edge_index, y=torch.tensor([cov_val], dtype=torch.float))

                graph_data_list.append(data)

        # Create DataLoader
        loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=True)

        return loader
    
    def process_graph_pair(self, tag, info):
        graph1 = info['graph1']
        graph2 = info['graph2']
        target = info['cov']

        adj1 = torch.tensor(nx.adjacency_matrix(graph1).toarray())
        adj2 = torch.tensor(nx.adjacency_matrix(graph2).toarray())

        x1 = adj1
        x2 = adj2

        edge_index = torch.cat([from_networkx(graph1).edge_index, from_networkx(graph2).edge_index + graph1.number_of_nodes()], dim=1)
        x = torch.cat([x1, x2], dim=0)

        data = Data(x=x, edge_index=edge_index, y=torch.tensor([target], dtype=torch.float))

        return data

    def create_graph_loader_parallel(self, batch_size: int = 1, num_cpus: int = None):
        if num_cpus is None:
            num_cpus = (os.cpu_count() - 1)  # Get the number of CPUs available

        graph_data_list = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = [executor.submit(self.process_graph_pair, tag, info) for tag, info in self.graph_data.items()]

            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                graph_data_list.append(data)

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
