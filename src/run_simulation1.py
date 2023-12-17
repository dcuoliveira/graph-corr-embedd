import os
import argparse
import numpy as np

from simulation.GraphSim import GraphSim
from utils.conn_data import save_pickle

parser = argparse.ArgumentParser()

parser.add_argument('--source_path', type=str, help='Source path for saving output.', default=os.path.dirname(__file__))
parser.add_argument('--simulation_name', type=str, help='Simulation name to be used on inputs dir.', default="simulation1")
parser.add_argument('--graph_name', type=str, help='Graph name to be generated.', default="erdos_renyi")

parser.add_argument('--n_simulations', type=int, help='Number of simulations.', default=30)
parser.add_argument('--n_graphs', type=int, help='Number of graphs per simulation.', default=50)
parser.add_argument('--n_nodes', type=int, help='Number of nodes.', default=100)

if __name__ == "__main__":
    args = parser.parse_args()

    # Check if path exists
    output_path = f"{args.source_path}/data/inputs/{args.simulation_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # generate covariance between gaussian random variables
    covs_xy = np.arange(-1, 1, 0.1)

    # init graph sim class
    gs = GraphSim(graph_name=args.graph_name)

    # start simulation procedure
    all_graphs = {}
    for s in covs_xy:
        for i in range(args.n_simulations):
            for j in range(args.n_graphs):

                # gen seed
                gs.update_seed()
                save_seed = gs.seed

                # generate probability of edge creation
                p = gs.get_p_from_bivariate_gaussian(s=s)

                # simulate graph
                graph1 = gs.simulate_erdos(n=args.n_nodes, prob=np.abs(p[0,0]))
                graph2 = gs.simulate_erdos(n=args.n_nodes, prob=np.abs(p[0,1]))

                # save graph
                graph_info = {
                    
                    "graph1": graph1,
                    "graph2": graph2,
                    "seed": save_seed,
                    "p": p,
                    "cov": s
                    
                }

                all_graphs[f"{np.round(s, 1)}_{i}_{j}"] = graph_info

    save_pickle(path=f"{output_path}/all_graphs.pkl", obj=all_graphs)


