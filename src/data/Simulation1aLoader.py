import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
import numpy as np

from utils.conn_data import load_pickle

class Simulation1aLoader(object):
    """
    Class to load data from simulation 1. The description of the simulation 1
    is given at Fujita et al. (2017) paper.

    References
    ----------
    Fujita, A. et al. (2017). Correlation between graphs with an application to brain network analysis.

    """
    
    def __init__(self, name: str="simulation1a", sample: bool=False):
        super().__init__()
    
        self.name = name
        self._read_data(sample=sample)

    def _read_data(self, sample: bool=False):
        if sample:
            self.graph_data = load_pickle(os.path.join(os.path.dirname(__file__), "inputs", self.name, "sample_graph_info.pkl"))
        else:
            self.graph_data = load_pickle(os.path.join(os.path.dirname(__file__), "inputs", self.name, "all_graph_info.pkl"))

    def create_graph_loader(self, batch_size: int=1):

        all_graph_data_dict = {}
        aux_query_list = []
        count = 0
        for i, (corr_tag, graph_list) in enumerate(self.graph_data.items()):

            corr_val = float(corr_tag.split("_")[0])
            n_nodes = int(corr_tag.split("_")[1])

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

                if corr_val != np.round(graph_pair_info["corr"], 1):
                    raise ValueError(f"Covariance value does not match: {corr_val}, {n_sim}")
                
                # Create a single Data object
                data = Data(x=x,
                            edge_index=edge_index,
                            y=torch.tensor([corr_val], dtype=torch.float),
                            n_nodes=n_nodes)

                all_graph_data_dict[count]= data

                aux_query_list.append(pd.DataFrame({"count": count, "corr": [corr_val], "n_nodes": [n_nodes]}))

                count += 1

        aux_query_df = pd.concat(aux_query_list, axis=0)
        loaders = {}
        for n_nodes in aux_query_df.n_nodes.unique():
            
            tmp_aux_query_df = aux_query_df[aux_query_df.n_nodes == n_nodes].reset_index(drop=True)
            n_nodes_loaders = []
            for idx, row in tmp_aux_query_df.iterrows():
                
                count = row['count']
                tmp_graph_data = all_graph_data_dict[count]
                n_nodes_loaders.append(tmp_graph_data)
            
            loader = DataLoader(n_nodes_loaders, batch_size=batch_size, shuffle=True)
            loaders[n_nodes] = loader

        return loaders

DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        import time

        start = time.time()
        sample = True
        dataset_name = "simulation1a"
        batch_size = 1

        sim = Simulation1aLoader(name=dataset_name, sample=sample)
        loader = sim.create_graph_loader(batch_size=batch_size)

        # time to minutes
        print("Time to load and process data: ", (time.time() - start) / 60)
