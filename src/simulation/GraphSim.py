import networkx as nx

class GraphSim:
    def __init__(self, graph_name: str, seed: int = 2294):
        """
        Random Graph simulation class.
        
        Args:
            graph_name: name of the graph to be generated.
            seed: random seed.
        
        Returns:
            None
        """
        self.graph_name = graph_name
        self.seed = seed

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
