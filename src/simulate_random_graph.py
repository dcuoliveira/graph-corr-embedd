import networkx as nx
import argparse

from simulation.GraphSim import GraphSim

parser = argparse.ArgumentParser()

parser.add_argument('--graph_name', type=str, help='graph name to be generated.', default="erdos_renyi")
parser.add_argument('--prob', type=float, help='prob. of edge creation.', default=0.5)
parser.add_argument('--n', type=int, help='number of nodes.', default=20)

if __name__ == "__main__":
    args = parser.parse_args()

    gs = GraphSim(graph_name=args.graph_name)
    graph_info = gs.simulate(n=args.n, prob=args.prob)