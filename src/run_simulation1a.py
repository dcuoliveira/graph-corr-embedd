import os
import numpy as np
from tqdm import tqdm

from simulation.GraphSim import GraphSim
from utils.conn_data import save_pickle
from utils.activation_functions import sigmoid

source_path = os.path.dirname(__file__)

def run_simulation1a(simulation_name: str, graph_name: str, sample: bool, n_simulations: int, n_graphs: int, n_nodes: int):

    # Check if path exists
    output_path = f"{source_path}/data/inputs/{simulation_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # generate covariance between gaussian random variables
    covs_xy = np.arange(-1, 1, 0.1)

    # init graph sim class
    gs = GraphSim(graph_name=graph_name)

    # check if sample graph
    if sample:
        covs_xy = covs_xy[:10]
        n_simulations = 2
        n_graphs = 1
        n_nodes = 100

    # start simulation procedure
    all_graphs = {}
    pbar = tqdm(covs_xy, total=len(covs_xy), desc=f"Simulating graphs for {simulation_name} with n_nodes={n_nodes}")
    for s in pbar:

        graphs_given_cov = []
        for i in range(n_simulations):

            # gen seed
            gs.update_seed()
            save_seed = gs.seed

            # generate probability of edge creation
            ps = gs.get_p_from_bivariate_gaussian(s=s, size=n_graphs)
            ps = sigmoid(ps)

            for j in range(n_graphs):

                # simulate graph
                graph1 = gs.simulate_erdos(n=n_nodes, prob=ps[j,0])
                graph2 = gs.simulate_erdos(n=n_nodes, prob=ps[j,1])

                # save graph
                sim_graph_info = {
                    
                    "n_simulations": i,
                    "n_graphs": j,
                    "graph1": graph1,
                    "graph2": graph2,
                    "seed": save_seed,
                    "p": ps[j,],
                    "corr": s # cov = corr becaus variances are 1
                    
                }

                graphs_given_cov.append(sim_graph_info)

        all_graphs[f"{np.round(s, 1)}"] = graphs_given_cov

    return all_graphs

