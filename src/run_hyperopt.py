from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from tqdm import tqdm
import torch
import torch.optim as optim
import argparse
import os
import random
import numpy as np
import pandas as pd

from models.SDNE import SDNE
from model_utils.EarlyStopper import EarlyStopper
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
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=True)
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1, choices=[1])
parser.add_argument('--model_name', type=str, help='Model name.', default="sdne9")
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Evaluate MSE on the test set
def evaluate_mse(test_list, model1, model2):
    mse_loss = torch.nn.MSELoss()
    model1.eval()
    model2.eval()
    
    mse1_tot, mse2_tot = 0, 0
    with torch.no_grad():
        for data in test_list:
            data = data.to(device)
            x1 = data.x[0, :, :].to(device)
            x2 = data.x[1, :, :].to(device)
            
            x1_hat, _, _ = model1.forward(x1)
            x2_hat, _, _ = model2.forward(x2)
            
            mse1 = mse_loss(x1_hat, x1)
            mse2 = mse_loss(x2_hat, x2)
            
            mse1_tot += mse1.item()
            mse2_tot += mse2.item()

    mse1_avg = mse1_tot / len(test_list)
    mse2_avg = mse2_tot / len(test_list)

    print(f"Test Set MSE for Model 1: {mse1_avg}")
    print(f"Test Set MSE for Model 2: {mse2_avg}")

    return mse1_avg, mse2_avg
    
space = {
    'alpha': hp.loguniform('alpha', np.log(1e-4), np.log(1e-1)),
    'beta': hp.loguniform('beta', np.log(1e0), np.log(1e2)),
    'gamma': hp.loguniform('gamma', np.log(1e2), np.log(1e4)),
    'nu': hp.loguniform('nu', np.log(1e-6), np.log(1e-4)),
}

args = parser.parse_args()

def objective(params):
    # Load dataset
    print('Loading the data from the simulation!')
    if args.dataset_name == "simulation1a":
        sim = Simulation1aLoader(name=args.dataset_na, sample=args.sample)
        dataset_list = sim.create_graph_list()

    elif args.dataset_name == "simulation1c":
        sim = Simulation1cLoader(name=args.dataset_name, sample=args.sample, graph_name=args.graph_name)
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

    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    nu = params['nu']
    
    # Initialize your models, optimizers, loss functions, etc.
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

    opt1 = optim.Adam(model1.parameters(), lr=args.learning_rate)
    opt2 = optim.Adam(model2.parameters(), lr=args.learning_rate)

    loss_local = LossLocal()
    loss_global = LossGlobal()
    loss_reg = LossReg()
    loss_eigen = LossEigen()

    early_stopper = EarlyStopper(patience=10, min_delta=100)

    epochs_tot_loss, val_tot_loss = [], []

    for epoch in range(args.epochs):
        batch_tot_loss1, batch_tot_loss2 = [], []
        for data in train_list:

            opt1.zero_grad()
            opt2.zero_grad()
            
            data = data.to(device)
            x1 = data.x[0, :, :].to(device)
            x2 = data.x[1, :, :].to(device)
            b1_mat, b2_mat = torch.ones_like(x1), torch.ones_like(x2)
            b1_mat[x1 != 0], b2_mat[x2 != 0] = beta, beta

            x1_hat, z1, z1_norm = model1.forward(x1)
            x2_hat, z2, z2_norm = model2.forward(x2)

            ll1 = loss_local.forward(adj=x1, z=z1)
            lg1 = loss_global.forward(adj=x1, x=x1_hat, b_mat=b1_mat)
            lr1 = loss_reg.forward(model=model1)

            lt1 = (alpha * lg1) + ll1 + (nu * lr1)
            batch_tot_loss1.append(lt1)

            ll2 = loss_local.forward(adj=x2, z=z2)
            lg2 = loss_global.forward(adj=x2, x=x2_hat, b_mat=b2_mat)
            lr2 = loss_reg.forward(model=model2)

            lt2 = (alpha * lg2) + ll2 + (nu * lr2)
            batch_tot_loss2.append(lt2)

            lt1.backward()
            opt1.step()
            lt2.backward()
            opt2.step()

            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)

        val_tot_loss1, val_tot_loss2 = [], []
        lt1_val_tot, lt2_val_tot = 0, 0
        with torch.no_grad():
            for data in val_list:

                data = data.to(device)
                x1 = data.x[0, :, :].to(device)
                x2 = data.x[1, :, :].to(device)
                b1_mat, b2_mat = torch.ones_like(x1), torch.ones_like(x2)
                b1_mat[x1 != 0], b2_mat[x2 != 0] = beta, beta

                x1_hat, z1, z1_norm = model1.forward(x1)
                x2_hat, z2, z2_norm = model2.forward(x2)

                ll1 = loss_local.forward(adj=x1, z=z1)
                lg1 = loss_global.forward(adj=x1, x=x1_hat, b_mat=b1_mat)
                lr1 = loss_reg.forward(model=model1)

                lt1 = (alpha * lg1) + ll1 + (nu * lr1)
                lt1_val_tot += lt1

                val_tot_loss1.append(lt1)

                ll2 = loss_local.forward(adj=x2, z=z2)
                lg2 = loss_global.forward(adj=x2, x=x2_hat, b_mat=b2_mat)
                lr2 = loss_reg.forward(model=model2)

                lt2 = (alpha * lg2) + ll2 + (nu * lr2)
                lt2_val_tot += lt2

                val_tot_loss2.append(lt2)

        if early_stopper.early_stop(lt1_val_tot) and early_stopper.early_stop(lt2_val_tot):             
            break

        val_tot_loss.append(torch.mean(torch.stack([lt1_val_tot, lt2_val_tot])))

    final_val_loss = val_tot_loss[-1].item()
    mse1, mse2 = evaluate_mse(test_list=test_list, model1=model1, model2=model2)

    return {
        'loss': final_val_loss,
        'mse1': mse1,
        'mse2': mse2,
        'status': STATUS_OK,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'nu': nu,
    }

# Run the hyperparameter search
trials = Trials()
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=20,  
    trials=trials
)

# Extract the best parameters
best_alpha = best_params['alpha']
best_beta = best_params['beta']
best_gamma = best_params['gamma']
best_nu = best_params['nu']

# Save the trial results
output_path = '.'
print('Saving: ', output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

save_pickle(path=f"{output_path}/hyperopt_trials.pkl", obj=trials)
save_pickle(path=f"{output_path}/best_params.pkl", obj=best_params)

results_df = pd.DataFrame(trials.results)
results_df.to_csv(f"{output_path}/hyperopt_results.csv", index=False)
