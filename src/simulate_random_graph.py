import os
import argparse
from simulation.GraphSim import GraphSim
from utils.conn_data import save_pickle

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--graph_name', type=str, help='Graph name to be generated.', default="erdos_renyi")
parser.add_argument('--n', type=int, help='Number of nodes.', default=20)
parser.add_argument('--seed', type=int, help='Random seed.', default=2294)
parser.add_argument('--simulations', type=int, help='Number of simulations.', default=1)
parser.add_argument('--source_path', type=str, help='Source path for saving output.', default=os.path.dirname(__file__))

# Erdos Renyi graph parameters
parser.add_argument('--prob', type=float, help='Probability of edge creation (for Erdos Renyi graph).', default=0.5)

# k-regular graph parameters
parser.add_argument('--k', type=int, help='Degree of each node (for k-regular graph).', default=3)

# Geometric graph parameters
parser.add_argument('--radius', type=float, help='Radius for edge creation (for geometric graph).', default=0.1)

# Barabasi Albert graph parameters
parser.add_argument('--m', type=int, help='Number of edges to attach from a new node to existing nodes (for Barabasi Albert graph).', default=1)

# Watts Strogatz graph parameters
parser.add_argument('--k_ws', type=int, help='Each node is connected to k nearest neighbors in ring topology (for Watts Strogatz graph).', default=4)
parser.add_argument('--p_ws', type=float, help='Probability of rewiring each edge (for Watts Strogatz graph).', default=0.1)

if __name__ == "__main__":
    args = parser.parse_args()

    gs = GraphSim(graph_name=args.graph_name, seed=args.seed)

    # Check if path exists
    output_path = f"{args.source_path}/data/outputs/{args.graph_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for i in range(args.simulations):
        if args.graph_name == "erdos_renyi":
            graph_info = gs.simulate_erdos(n=args.n, prob=args.prob)
            save_pickle(path=f"{output_path}/graph_info_{args.n}_{args.prob}_sim_{i}.pkl", obj=graph_info)
        elif args.graph_name == "k_regular":
            graph_info = gs.simulate_k_regular(n=args.n, k=args.k)
            save_pickle(path=f"{output_path}/graph_info_{args.n}_{args.k}_sim_{i}.pkl", obj=graph_info)
        elif args.graph_name == "geometric":
            graph_info = gs.simulate_geometric(n=args.n, radius=args.radius)
            save_pickle(path=f"{output_path}/graph_info_{args.n}_{args.radius}_sim_{i}.pkl", obj=graph_info)
        elif args.graph_name == "barabasi_albert":
            graph_info = gs.simulate_barabasi_albert(n=args.n, m=args.m)
            save_pickle(path=f"{output_path}/graph_info_{args.n}_{args.m}_sim_{i}.pkl", obj=graph_info)
        elif args.graph_name == "watts_strogatz":
            graph_info = gs.simulate_watts_strogatz(n=args.n, k=args.k_ws, p=args.p_ws)
            save_pickle(path=f"{output_path}/graph_info_{args.n}_{args.k_ws}_{args.p_ws}_sim_{i}.pkl", obj=graph_info)
        else:
            raise ValueError("Invalid graph name")
