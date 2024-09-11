

import torch
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from torch_geometric.data import DataLoader
from model_utils.EarlyStopper import EarlyStopper
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd

from models.SDNE import SDNE
from data.Simulation1aLoader import Simulation1aLoader
from data.Simulation1cLoader import Simulation1cLoader
from loss_functions.LossGlobal import LossGlobal
from loss_functions.LossLocal import LossLocal
from loss_functions.LossReg import LossReg
from loss_functions.LossEigen import LossEigen

from utils.conn_data import save_pickle
from utils.parsers import str_2_bool

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="simulation1c")
parser.add_argument('--graph_name', type=str, help='Graph name.', default="erdos_renyi")
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=False)
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1, choices=[1])
parser.add_argument('--model_name', type=str, help='Model name.', default="sdne8oldes")
parser.add_argument('--n_nodes', type=int, help='Number of nodes.', default=100)
parser.add_argument('--shuffle', type=str, help='Shuffle the dataset.', default=True)
parser.add_argument('--epochs', type=int, help='Epochs to train the model.', default=10)

parser.add_argument('--n_hidden', type=int, help='Number of hidden dimensions in the nn.', default=100)
parser.add_argument('--n_layers_enc', type=int, help='Number of layers in the encoder network.', default=1)
parser.add_argument('--n_layers_dec', type=int, help='Number of layers in the decoder network.', default=1)
parser.add_argument('--dropout', type=float, help='Dropout rate (1 - keep probability).', default=0.5)
parser.add_argument('--learning_rate', type=float, help='Learning rate of the optimization algorithm.', default=0.001)
parser.add_argument('--early_stopping', type=bool, default=True, help='Bool to specify if to use early stopping.')
parser.add_argument('--gradient_clipping', type=bool, default=True, help='Bool to specify if to use gradient clipping.')
parser.add_argument('--stadardize_losses', type=bool, default=False, help='Bool to specify if to standardize the value of loss functions.')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameter search space
space = {
    'alpha': hp.uniform('alpha', float(1e-5), 1),
    'beta': hp.uniform('beta', 1e-5, 1),
    'theta': hp.uniform('theta', 1e-5, 1),
    'nu': hp.uniform('nu', 1e-5, 1),
    'gamma': hp.uniform('gamma', 1e-5, 1),
}

def objective(params):
    # parse args
    args = parser.parse_args()

    # convert to boolean
    args.sample = str_2_bool(args.sample)
    args.shuffle = str_2_bool(args.shuffle)
    args.early_stopping = str_2_bool(args.early_stopping)
    args.gradient_clipping = str_2_bool(args.gradient_clipping)

    # define dataset
    print('Loading the data from the simulation!')
    if args.dataset_name == "simulation1a":
        sim = Simulation1aLoader(name=args.dataset_name, sample=args.sample)
        dataset_list = sim.create_graph_list()
    elif args.dataset_name == "simulation1c":
        sim = Simulation1cLoader(name=args.dataset_name, sample=args.sample, graph_name = args.graph_name)
        print('Loading the simulation data!')
        dataset_list = sim.create_graph_list(load_preprocessed=True)
    else:
        raise Exception('Dataset not found!')
    print('Finish Loading')

    # Train, validation, test split
    n = len(dataset_list)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size
    
    train_idx = np.random.randint(0, n, size=train_size)
    train_set = set(train_idx)
    all_indices = set(range(n))
    available_indices = all_indices - train_set
    available_indices = list(available_indices)
    val_idx = np.random.choice(available_indices, size=val_size, replace=False)
    available_indices = all_indices - set(val_idx) - train_set
    test_idx = np.array(list(available_indices))
    
    train_list = [dataset_list[i] for i in train_idx]
    val_list = [dataset_list[i] for i in val_idx]
    test_list = [dataset_list[i] for i in test_idx]

    # define early stopper
    early_stopper = EarlyStopper(patience=10, min_delta=100)

    # define model
    model1 = SDNE(node_size=args.n_nodes,
                  n_hidden=args.n_hidden,
                  n_layers_enc=args.n_layers_enc,
                  n_layers_dec=args.n_layers_dec,
                  bias_enc=True,
                  bias_dec=True,
                  droput=args.dropout).to(device)
    
    model2 = SDNE(node_size=args.n_nodes,
                  n_hidden=args.n_hidden,
                  n_layers_enc=args.n_layers_enc,
                  n_layers_dec=args.n_layers_dec,
                  bias_enc=True,
                  bias_dec=True,
                  droput=args.dropout).to(device)
    
    # define optimizer
    opt1 = optim.Adam(model1.parameters(), lr=args.learning_rate)
    opt2 = optim.Adam(model2.parameters(), lr=args.learning_rate)

    # define loss functions
    loss_local = LossLocal()
    loss_global = LossGlobal()
    loss_reg = LossReg()
    loss_eigen = LossEigen()

    # initialize tqdm
    pbar = tqdm(range(args.epochs), total=args.epochs, desc=f"Running {args.model_name} model")
    epochs_tot_loss, epochs_global_loss, epochs_local_loss, epochs_reg_loss, epochs_eigen_loss = [], [], [], [], []
    epochs_predictions = []

    # SDNE TRAINING
    for epoch in pbar:
        opt1.zero_grad()
        opt2.zero_grad()

        epoch_results = []
        batch_tot_loss1, batch_global_loss1, batch_local_loss1, batch_reg_loss1, batch_eigen_loss1 = [], [], [], [], []
        batch_tot_loss2, batch_global_loss2, batch_local_loss2, batch_reg_loss2, batch_eigen_loss2 = [], [], [], [], []
        batch_predictions = []
        for data in train_list:
            data = data.to(device)

            # get inputs
            x1 = data.x[0, :, :].to(device)
            x2 = data.x[1, :, :].to(device)

            # create global loss parameter matrix
            b1_mat, b2_mat = torch.ones_like(x1), torch.ones_like(x2)
            b1_mat[x1 != 0], b2_mat[x2 != 0] = params['beta'], params['beta']

            # forward pass
            x1_hat, z1, z1_norm = model1.forward(x1)
            x2_hat, z2, z2_norm = model2.forward(x2)

            # compute correlation between embeddings (true target)
            pred_cov = model1.compute_spearman_rank_correlation(x=z1.flatten().detach(), y=z2.flatten().detach())

            # store pred and true values
            batch_predictions.append([pred_cov, data.y])

            # compute loss functions I
            ll1 = loss_local.forward(adj=x1, z=z1)
            lg1 = loss_global.forward(adj=x1, x=x1_hat, b_mat=b1_mat)
            lr1 = loss_reg.forward(model=model1)
            le1 = loss_eigen.forward(adj=x1, x=x1_hat)

            if args.stadardize_losses:
                l1_sum = lg1.item() + ll1.item() + lr1.item() + le1.item()
                lg1 /= l1_sum
                ll1 /= l1_sum
                lr1 /= l1_sum
                le1 /= l1_sum
            lt1 = (params['alpha'] * lg1) + (params['theta'] * ll1) + (params['nu'] * lr1) + (params['gamma'] * le1)

            batch_tot_loss1.append(lt1.item())
            batch_global_loss1.append(lg1.item())
            batch_local_loss1.append(ll1.item())
            batch_reg_loss1.append(lr1.item())
            batch_eigen_loss1.append(le1.item())

            # compute loss functions II
            ll2 = loss_local.forward(adj=x2, z=z2)
            lg2 = loss_global.forward(adj=x2, x=x2_hat, b_mat=b2_mat)
            lr2 = loss_reg.forward(model=model2)
            le2 = loss_eigen.forward(adj=x2, x=x2_hat)

            if args.stadardize_losses:
                l2_sum = lg2.item() + ll2.item() + lr2.item() + le2.item()
                lg2 /= l2_sum
                ll2 /= l2_sum
                lr2 /= l2_sum
                le2 /= l2_sum
            lt2 = (params['alpha'] * lg2) + (params['theta'] * ll2) + (params['nu'] * lr2) + (params['gamma'] * le2)

            batch_tot_loss2.append(lt2.item())
            batch_global_loss2.append(lg2.item())
            batch_local_loss2.append(ll2.item())
            batch_reg_loss2.append(lr2.item())
            batch_eigen_loss2.append(le2.item())

            # backward pass
            lt1.backward()
            lt2.backward()

        opt1.step()
        opt2.step()

        # gradient clipping
        if args.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)

        # Validation
        val_tot_loss1, val_tot_loss2 = [], []
        with torch.no_grad():
            for data in val_list:
                data = data.to(device)
                x1 = data.x[0, :, :].to(device)
                x2 = data.x[1, :, :].to(device)
                b1_mat, b2_mat = torch.ones_like(x1), torch.ones_like(x2)
                b1_mat[x1 != 0], b2_mat[x2 != 0] = params['beta'], params['beta']

                x1_hat, z1, z1_norm = model1.forward(x1)
                x2_hat, z2, z2_norm = model2.forward(x2)

                ll1 = loss_local.forward(adj=x1, z=z1)
                lg1 = loss_global.forward(adj=x1, x=x1_hat, b_mat=b1_mat)
                lr1 = loss_reg.forward(model=model1)
                le1 = loss_eigen.forward(adj=x1, x=x1_hat)

                lt1 = (params['alpha'] * lg1) + (params['theta'] * ll1) + (params['nu'] * lr1) + (params['gamma'] * le1)
                val_tot_loss1.append(lt1.item())

                ll2 = loss_local.forward(adj=x2, z=z2)
                lg2 = loss_global.forward(adj=x2, x=x2_hat, b_mat=b2_mat)
                lr2 = loss_reg.forward(model=model2)
                le2 = loss_eigen.forward(adj=x2, x=x2_hat)

                lt2 = (params['alpha'] * lg2) + (params['theta'] * ll2) + (params['nu'] * lr2) + (params['gamma'] * le2)
                val_tot_loss2.append(lt2.item())

        # early stopping
        if args.early_stopping:
            if early_stopper.early_stop(np.mean(val_tot_loss1)) and early_stopper.early_stop(np.mean(val_tot_loss2)):             
                break

        epochs_predictions.append(torch.tensor(batch_predictions).to(device))
        epochs_tot_loss.append(torch.tensor([np.mean(batch_tot_loss1), np.mean(batch_tot_loss2)]).to(device))
        epochs_global_loss.append(torch.tensor([np.mean(batch_global_loss1), np.mean(batch_global_loss2)]).to(device))
        epochs_local_loss.append(torch.tensor([np.mean(batch_local_loss1), np.mean(batch_local_loss2)]).to(device))
        epochs_reg_loss.append(torch.tensor([np.mean(batch_reg_loss1), np.mean(batch_reg_loss2)]).to(device))
        epochs_eigen_loss.append(torch.tensor([np.mean(batch_eigen_loss1), np.mean(batch_eigen_loss2)]).to(device))

    # Evaluate on test set
    test_tot_loss1, test_tot_loss2 = [], []
    with torch.no_grad():
        for data in test_list:
            data = data.to(device)
            x1 = data.x[0, :, :].to(device)
            x2 = data.x[1, :, :].to(device)
            b1_mat, b2_mat = torch.ones_like(x1), torch.ones_like(x2)
            b1_mat[x1 != 0], b2_mat[x2 != 0] = params['beta'], params['beta']

            x1_hat, z1, z1_norm = model1.forward(x1)
            x2_hat, z2, z2_norm = model2.forward(x2)

            ll1 = loss_local.forward(adj=x1, z=z1)
            lg1 = loss_global.forward(adj=x1, x=x1_hat, b_mat=b1_mat)
            lr1 = loss_reg.forward(model=model1)
            le1 = loss_eigen.forward(adj=x1, x=x1_hat)

            lt1 = (params['alpha'] * lg1) + (params['theta'] * ll1) + (params['nu'] * lr1) + (params['gamma'] * le1)
            test_tot_loss1.append(lt1.item())

            ll2 = loss_local.forward(adj=x2, z=z2)
            lg2 = loss_global.forward(adj=x2, x=x2_hat, b_mat=b2_mat)
            lr2 = loss_reg.forward(model=model2)
            le2 = loss_eigen.forward(adj=x2, x=x2_hat)

            lt2 = (params['alpha'] * lg2) + (params['theta'] * ll2) + (params['nu'] * lr2) + (params['gamma'] * le2)
            test_tot_loss2.append(lt2.item())

    final_test_loss = np.mean(test_tot_loss1 + test_tot_loss2)
    final_val_loss = np.mean(val_tot_loss1 + val_tot_loss2)

    return {
        'loss': final_val_loss,
        'test_loss': final_test_loss,
        'status': STATUS_OK,
        'params': params
    }

if __name__ == '__main__':
    # Run the hyperparameter search
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )

    # Extract the best parameters
    best_params = {
        'alpha': best['alpha'],
        'beta': best['beta'],
        'theta': best['theta'],
        'nu': best['nu'],
        'gamma': best['gamma']
    }

    # Save the trial results
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{args.graph_name}/{args.model_name}_hyperopt"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    save_pickle(path=f"{output_path}/hyperopt_trials.pkl", obj=trials)
    save_pickle(path=f"{output_path}/best_params.pkl", obj=best_params)

    results_df = pd.DataFrame(trials.results)
    results_df.to_csv(f"{output_path}/hyperopt_results.csv", index=False)

    print("Best hyperparameters found:")
    print(best_params)
    print(f"Best validation loss: {trials.best_trial['result']['loss']}")
    print(f"Corresponding test loss: {trials.best_trial['result']['test_loss']}")
