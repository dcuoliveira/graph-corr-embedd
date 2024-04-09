import networkx as nx
import numpy as np

class GraphSim:
    def __init__(self, graph_name: str):
        """
        Random Graph simulation class.
        
        Args:
            graph_name: name of the graph to be generated.
            seed: random seed.
        
        Returns:
            None
        """
        self.graph_name = graph_name

    def get_p_from_bivariate_gaussian(self, s: float, size: int):
        """
        Generate probability of edge creation from a bivariate gaussian distribution.
        
        Args:
            s: covariance between gaussian random variables.
            size: size of the output matrix.
            
        Returns:
            probability of edge creation.
        """

        p = np.random.multivariate_normal(mean=[0, 0], cov=[[1, s], [s, 1]], size=size)

        return p

    def update_seed(self, seed: int=None):
        """
        Update random seed.
        
        Args:
            seed: random seed.
            
        Returns:
            None
        """
        self.seed = np.random.randint(0, 100000) if seed is not None else seed

    def simulate_erdos(self, n: int, prob: float):
        """
        Simulate a random Erdős-Rényi graph.
        
        Args:
            n: number of nodes.
            prob: probability of edge creation.
            
        Returns:
            networkx graph object.
        """
        return nx.erdos_renyi_graph(n=n, p=prob, seed=self.seed)

    def simulate_k_regular(self, n: int, k: int):
        """
        Simulate a random k-regular graph.
        
        Args:
            n: number of nodes.
            k: degree of each node.
            
        Returns:
            networkx graph object.
        """
        return nx.random_regular_graph(d=k, n=n, seed=self.seed)

    def simulate_geometric(self, n: int, radius: float):
        """
        Simulate a random geometric graph.
        
        Args:
            n: number of nodes.
            radius: radius for edge creation.
            
        Returns:
            networkx graph object.
        """
        return nx.random_geometric_graph(n=n, radius=radius, seed=self.seed)

    def simulate_barabasi_albert(self, n: int, m: int):
        """
        Simulate a random Barabási-Albert preferential attachment graph.
        
        Args:
            n: number of nodes.
            m: number of edges to attach from a new node to existing nodes.
            
        Returns:
            networkx graph object.
        """
        return nx.barabasi_albert_graph(n=n, m=m, seed=self.seed)

    def simulate_watts_strogatz(self, n: int, k: int, p: float):
        """
        Simulate a random Watts-Strogatz small-world graph.
        
        Args:
            n: number of nodes.
            k: each node is joined with its k nearest neighbors in a ring topology.
            p: probability of rewiring each edge.
            
        Returns:
            networkx graph object.
        """
        return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=self.seed)
