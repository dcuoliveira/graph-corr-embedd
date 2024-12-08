
import numpy as np
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram

from src.stats.Stats import Stats

class WL(Stats):
    def __init__(self, n_iter=3):
        super(WL, self).__init__()
        self.n_iter = n_iter
        self.wl_kernel = WeisfeilerLehman(
            n_iter=n_iter, 
            normalize=True, 
            base_graph_kernel=VertexHistogram
        )

    def forward(self, adj1, adj2):
        # Initialize node attributes (all nodes have label 1)
        nodes_attributes = {n: 1 for n in range(adj1.shape[0])}
        
        # Create grakel Graph objects
        G1 = Graph(adj1, node_labels=nodes_attributes)
        G2 = Graph(adj2, node_labels=nodes_attributes)
        
        # Compute kernel matrix
        K = self.wl_kernel.fit_transform([G1, G2])
        
        # Return distance (inverse of kernel)
        return 1.0 / (1 + K[0,1])

if __name__ == "__main__":
    pass