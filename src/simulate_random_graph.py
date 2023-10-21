import os
import argparse

from src.simulation.GraphSim import GraphSim
from src.utils.conn_data import save_pickle

parser = argparse.ArgumentParser()

parser.add_argument('--graph_name', type=str, help='graph name to be generated.', default="erdos_renyi")
parser.add_argument('--prob', type=float, help='prob. of edge creation.', default=0.5)
parser.add_argument('--n', type=int, help='number of nodes.', default=20)
parser.add_argument('--seed', type=int, help='random seed.', default=2294)
parser.add_argument('--source_path', type=str, help='random seed.', default=os.path.dirname(__file__))

if __name__ == "__main__":
    args = parser.parse_args()

    gs = GraphSim(graph_name=args.graph_name, seed=args.seed)
    graph_info = gs.simulate(n=args.n, prob=args.prob)

    # check if path exists
    if not os.path.exists(f"{args.source_path}/data/outputs/{args.graph_name}"):
        os.makedirs(f"{args.source_path}/data/outputs/{args.graph_name}")

    # save graph info
    save_pickle(path=f"{args.source_path}/data/outputs/{args.graph_name}/graph_info_{args.prob}_{args.n}.pkl", obj=graph_info)
