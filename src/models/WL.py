import numpy as np
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram

from src.stats.Stats import Stats

class WL(Stats):
    """
    Weisfeiler-Lehman Graph Distance Calculator
    
    This class computes a distance between two graphs using the Weisfeiler-Lehman (WL) kernel.
    The WL kernel works by:
    1. Initially labeling all nodes with the same label (1)
    2. Iteratively updating node labels based on neighbor labels for n_iter iterations
    3. Computing a kernel value based on the histogram of labels at each iteration
    
    The final distance is computed as 1/(1 + K) where K is the kernel value.
    - When graphs are very similar, K is close to 1, giving a distance close to 0.5
    - When graphs are very different, K is close to 0, giving a distance close to 1
    """
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
        
        # Compute kernel matrix - higher values mean more similar graphs
        K = self.wl_kernel.fit_transform([G1, G2])
        
        # Return distance (inverse of kernel)
        # As K increases (more similar), distance decreases
        # As K decreases (less similar), distance increases
        return 1.0 / (1 + K[0,1])

if __name__ == "__main__":
    pass