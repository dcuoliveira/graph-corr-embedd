import os
import argparse
import numpy as np
from tqdm import tqdm

from simulation.GraphSim import GraphSim
from utils.conn_data import save_pickle
from utils.activation_functions import sigmoid

parser = argparse.ArgumentParser()

parser.add_argument('--source_path', type=str, help='Source path for saving output.', default=os.path.dirname(__file__))
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=False)
parser.add_argument('--simulation_name', type=str, help='Simulation name to be used on inputs dir.', default="simulation2a")
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
    graph_classes = [
        "erdos_renyi",
        "random_geometric",
        "random_regular",
        "barabasi_albert",
        "watts_strogatz",
    ]

    # start simulation procedure
    all_graphs = {}
    for graph_name in tqdm(graph_classes, total=len(graph_classes), desc=f"Simulating graphs for {args.simulation_name}"):

        # init graph sim class
        gs = GraphSim(graph_name=graph_name)

        graphs_given_cov = []
        for i in range(args.n_simulations):

            for j in range(args.n_graphs):

                # gen seed
                gs.update_seed()
                save_seed = gs.seed

                # generate probability of edge creation
                p = gs.get_p_from_bivariate_gaussian(s=0)
                p = sigmoid(p.__abs__())

                if graph_name == "erdos_renyi":
                    graph1 = gs.simulate_erdos(n=args.n_nodes, prob=p[0,0])
                    graph2 = gs.simulate_erdos(n=args.n_nodes, prob=p[0,1])
                elif graph_name == "random_geometric":
                    graph1 = gs.simulate_random_geometric(n=args.n_nodes, prob=p[0,0])
                    graph2 = gs.simulate_random_geometric(n=args.n_nodes, prob=p[0,1])

                # save graph
                sim_graph_info = {
                    
                    "n_simulations": i,
                    "n_graphs": j,
                    "graph1": graph1,
                    "graph2": graph2,
                    "seed": save_seed,
                    "p": p,
                    "cov": s
                    
                }

                graphs_given_cov.append(sim_graph_info)

        all_graphs[f"{np.round(s, 1)}"] = graphs_given_cov

    if not args.sample:
        save_pickle(path=f"{output_path}/all_graph_info.pkl", obj=all_graphs)
    else:
        save_pickle(path=f"{output_path}/sample_graph_info.pkl", obj=all_graphs)

