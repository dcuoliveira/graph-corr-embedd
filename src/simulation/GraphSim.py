import networkx as nx

class GraphSim:
    def __init__(self, graph_name: str):
        """
        Random Graph simulation class.
        
        Args:
            graph_name: name of the graph to be generated.
        
        Returns:
            None
        """


        self.graph_name = graph_name

    def simulate(self, n: int, prob: float):
        """"
        Simulate a random graph.
        
        Args:
            n: number of nodes.
            prob: probability of edge creation.
            
        Returns:
            networkx graph object.
        """
        
        if self.graph_name == "erdos_renyi":
            graph = nx.erdos_renyi_graph(n=n, p=prob)
        else:
            raise ValueError("Graph name not supported.")
        
        return graph

