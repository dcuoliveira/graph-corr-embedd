
import os
import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, norm
import networkx as nx
import matplotlib.pyplot as plt

from simulation.GraphSim import GraphSim
from utils.conn_data import save_pickle
from utils.activation_functions import sigmoid

import torch
from models.Spectrum import Spectrum
from models.SAE import StackedSparseAutoencoder
from models.SDNE import SDNE

parser = argparse.ArgumentParser()

parser.add_argument('--source_path', type=str, help='Source path for saving output.', default=os.path.dirname(__file__))
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=False)
parser.add_argument('--simulation_name', type=str, help='Simulation name to be used on inputs dir.', default="simulation2a")
#parser.add_argument('--graph_types', type=np.array, help='Graph name to be generated.', default=["erdos_renyi", "random_geometric", "watts_strogatz"])
parser.add_argument('--n_simulations', type=int, help='Number of simulations.', default=30)
#parser.add_argument('--n_graphs', type=np.array, help='Number of graphs per simulation.', default=[10,20])
parser.add_argument('--n_nodes', type=int, help='Number of nodes.', default=100)
parser.add_argument('--covariance', type=float, help='Covariance.', default=0)

def get_spearman_pvalues(eigenvalues_dict, first_family_name, second_family_name, n_simulations, n_graph):
    pvalues = []
    for i in range(n_simulations):
        first_vector = np.array(eigenvalues_dict[first_family_name][second_family_name][n_graph][i][0])
        second_vector = np.array(eigenvalues_dict[first_family_name][second_family_name][n_graph][i][1])
        pvalues.append(spearmanr(first_vector.flatten(), second_vector.flatten()).pvalue)
    return pvalues

def simulation_graph_case(gs, theta, graph_name, n_nodes):
    if graph_name == "erdos_renyi":
        graph = gs.simulate_erdos(n=n_nodes, prob=theta)
    elif graph_name == "random_geometric":
        graph = gs.simulate_geometric(n=n_nodes, radius=theta)
    elif graph_name == "k_regular":
        if n_nodes * int(10*theta) % 2 != 0:
            graph = gs.simulate_k_regular(n=n_nodes, k=int(10*theta)+1)
        else:
            graph = gs.simulate_k_regular(n=n_nodes, k=int(10*theta))
    elif graph_name == "barabasi_albert":
        if int(10*theta) == 0:
            graph = gs.simulate_barabasi_albert(n=n_nodes, m=1)
        else:
            graph = gs.simulate_barabasi_albert(n=n_nodes, m=int(10*theta))
    elif graph_name == "watts_strogatz":
        graph = gs.simulate_watts_strogatz(n=n_nodes, k=3, p=theta)
    return graph

def get_dicts(n_graphs, graph_classes):
    embed_dict = {graph_name: {n_graph: [] for n_graph in n_graphs} for graph_name in graph_classes}
    embed_dict = {graph_name: embed_dict for graph_name in graph_classes}
    params_dict = {graph_name: {n_graph: [] for n_graph in n_graphs} for graph_name in graph_classes}
    params_dict = {graph_name: params_dict for graph_name in graph_classes}
    return embed_dict, params_dict

#TODO: Fix
def load_model(model_name, graph1, graph2):
    if model_name == 'SAE':
        path = 'src/data/outputs/simulation1/sae_[30, 15, 30]_1000/model1_sample.pth'
        model1 = StackedSparseAutoencoder(input_size=100, hidden_sizes=[30,15,30], dropout=0.5, sparsity_penalty=1e-4)  # initialize the model first
        model1.load_state_dict(torch.load(path))  # replace with your actual path
        model1.eval()  # set the model to evaluation mode

        path = 'src/data/outputs/simulation1/sae_[30, 15, 30]_1000/model2_sample.pth'
        model2 = StackedSparseAutoencoder(input_size=100, hidden_sizes=[30,15,30], dropout=0.5, sparsity_penalty=1e-4)  # initialize the model first
        model2.load_state_dict(torch.load(path))  # replace with your actual path
        model2.eval()  # set the model to evaluation mode


        input1 = torch.tensor(nx.adjacency_matrix(graph1).A, dtype=torch.float32)
        input2 = torch.tensor(nx.adjacency_matrix(graph2).A, dtype=torch.float32)

        with torch.no_grad():
            _, largest_eigenvalue1 = model1.forward(input1)
            _, largest_eigenvalue2 = model2.forward(input2)
            largest_eigenvalue1 = largest_eigenvalue1.detach().flatten()
            largest_eigenvalue2 = largest_eigenvalue2.detach().flatten()

    elif model_name=='eigenvalue':
        model1 = Spectrum()
        model2 = Spectrum()
        input1 = nx.adjacency_matrix(graph1).A
        input2 = nx.adjacency_matrix(graph2).A
        largest_eigenvalue1 = model1.forward(input1).flatten()
        largest_eigenvalue2 = model2.forward(input2).flatten()

    elif model_name=='sdne':
        pass
    elif model_name=='siamese':
        pass
    else:
        pass

    return largest_eigenvalue1, largest_eigenvalue1

def simulate_vector_graphs(n_graph, graph_name1, graph_name2, n_nodes, covariance):
    embeddings1, embeddings2 = [], []
    thetas1, thetas2 = [], []

    gs = GraphSim(graph_name=graph_name1)  # It doesn't matter the graph_name here
    gs.update_seed()
    ps = gs.get_p_from_bivariate_gaussian(s=covariance, size=n_graph)
    ps = sigmoid(ps)  # normalize

    for j in range(n_graph):
        p1, p2 = ps[j, 0], ps[j, 1]

        # Generate parameters and normalize then gen graph
        graph1  = simulation_graph_case(gs, p1, graph_name1, n_nodes=n_nodes)
        graph2  = simulation_graph_case(gs, p2, graph_name2, n_nodes=n_nodes)

        ##############
        largest_eigenvalue1, largest_eigenvalue2 = load_model('SAE', graph1, graph2)
        #largest_eigenvalue1, largest_eigenvalue2 = load_model('eigenvalue', graph1, graph2)
        ##############

        embeddings1.append(largest_eigenvalue1)
        embeddings2.append(largest_eigenvalue2)

        thetas1.append(p1)
        thetas2.append(p2)

    return embeddings1, embeddings2, thetas1, thetas2

def ensure_nested_dicts(embed_dict, params_dict, graph_name1, graph_name2, n_graph):
    if graph_name1 not in embed_dict:
        embed_dict[graph_name1] = {}
    if graph_name2 not in embed_dict[graph_name1]:
        embed_dict[graph_name1][graph_name2] = {}
    if n_graph not in embed_dict[graph_name1][graph_name2]:
        embed_dict[graph_name1][graph_name2][n_graph] = []
    
    if graph_name1 not in params_dict:
        params_dict[graph_name1] = {}
    if graph_name2 not in params_dict[graph_name1]:
        params_dict[graph_name1][graph_name2] = {}
    if n_graph not in params_dict[graph_name1][graph_name2]:
        params_dict[graph_name1][graph_name2][n_graph] = []
    
    return embed_dict, params_dict

def run_simulation(n_graphs, n_simulations, n_nodes, covariance, graph_classes):
    embed_dict, params_dict = get_dicts(n_graphs, graph_classes)
    simulated_combinations = set()

    for n_graph in n_graphs:
        for graph_name1 in tqdm(graph_classes, desc=f"Simulating {n_graph} graphs"):
            for graph_name2 in tqdm(graph_classes, desc=f"Simulating {n_graph} graphs"):
                
                # Check if this combination or its reverse has been simulated
                if (graph_name1, graph_name2, n_graph) in simulated_combinations or (graph_name2, graph_name1, n_graph) in simulated_combinations:
                    continue  # Skip this combination if it has already been simulated
                
                # Mark this combination as simulated
                simulated_combinations.add((graph_name1, graph_name2, n_graph))
                for _ in range(n_simulations):
                    embeddings1, embeddings2, thetas1, thetas2 = simulate_vector_graphs(n_graph, graph_name1, graph_name2,
                                                                                        n_nodes, covariance=covariance)

                    embed_dict, params_dict = ensure_nested_dicts(embed_dict=embed_dict, params_dict=params_dict, graph_name1=graph_name1,
                                                                  graph_name2=graph_name2, n_graph=n_graph)

                    # Store the results
                    embed_dict[graph_name1][graph_name2][n_graph].append((embeddings1, embeddings2))
                    params_dict[graph_name1][graph_name2][n_graph].append((thetas1, thetas2))

    return embed_dict, params_dict

def plot_roc_curves(graph_types, n_graphs, eigen, params, n_simulations):
    def calculate_rates(p_values, threshold):
        fp, tn = 0, 0
        for p_value in p_values:
            if p_value > threshold:
                fp += 1
            else:
                tn += 1
        return tn / (fp + tn) if (fp + tn) > 0 else 0

    fig, axs = plt.subplots(len(graph_types), len(graph_types), figsize=(15, 15))
    for n_graph in n_graphs:
        for i, gt1 in enumerate(graph_types):
            for j, gt2 in enumerate(graph_types):
                if j >= i:  # This condition ensures we only fill the upper triangle
                    pval = get_spearman_pvalues(eigen, n_simulations=n_simulations, n_graph=n_graph,
                                                first_family_name=gt1, second_family_name=gt2)
                    thresholds = np.linspace(0, 1, len(pval))
                    fprs = [calculate_rates(pval, th) for th in thresholds]
                    
                    axs[i, j].plot(thresholds, fprs, marker='.', label=f'{n_graph}')
                    axs[i, j].plot([0, 1], [0, 1], linestyle='--')
                    axs[i, j].set_xlabel('P-value Threshold')
                    axs[i, j].set_title(f'{gt1} vs {gt2}')
                    axs[i, j].legend()
                else:
                    axs[i, j].axis('off')

    pval_baseline = get_spearman_pvalues(params, n_simulations=n_simulations, n_graph=n_graph,
                                        first_family_name='erdos_renyi', second_family_name='erdos_renyi')
    fprs_baseline = [calculate_rates(pval_baseline, th) for th in thresholds]
    axs[-1, 0].axis('on')
    axs[-1, 0].plot(thresholds, fprs_baseline, marker='.', label=f'{n_graphs[-1]}')
    axs[-1, 0].plot([0, 1], [0, 1], linestyle='--')
    axs[-1, 0].set_xlabel('P-value Threshold')
    axs[-1, 0].set_title('Baseline')
    axs[-1, 0].legend()
    plt.suptitle('Upper Triangle of ROC Curves Matrix', fontsize=16)
    plt.tight_layout()
    plt.legend()
    plt.close()
    return fig

if __name__ == "__main__":
    args = parser.parse_args()
    args.graph_types = [
            "erdos_renyi",
            "random_geometric",
            #"k_regular",
            #"barabasi_albert",
            #"watts_strogatz",
        ]
    args.n_graphs = [10, 20]
    #args.n_graphs = [20, 40, 60, 80, 100]

    # Check if path exists
    input_path = f"{args.source_path}/data/inputs/{args.simulation_name}"
    output_path = f"{args.source_path}/data/outputs/{args.simulation_name}"
    print(input_path)
    print(output_path)

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Run the simulation
    eigen, params = run_simulation(args.n_graphs, args.n_simulations, args.n_nodes, args.covariance, args.graph_types)
    
    if not args.sample:
        save_pickle(path=f"{input_path}/eigenvalues_graphs_info.pkl", obj=eigen)
        save_pickle(path=f"{input_path}/parameters_graphs_info.pkl", obj=params)
    else:
        save_pickle(path=f"{input_path}/sample_eigenvalues_graphs_info.pkl", obj=eigen)
        save_pickle(path=f"{input_path}/sample_parameters_graphs_info.pkl", obj=params)

    fig = plot_roc_curves(args.graph_types, args.n_graphs, eigen, params, args.n_simulations)
    fig.savefig(os.path.join(output_path, 'result.png'))


