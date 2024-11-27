import torch
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from torch_geometric.data import DataLoader
from model_utils.EarlyStopper import EarlyStopper
import numpy as np

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
parser.add_argument('--graph_name', type=str, help='Graph name.', default="watts_strogatz")
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=True)
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1, choices=[1])
parser.add_argument('--model_name', type=str, help='Model name.', default="sdne8old")
parser.add_argument('--n_nodes', type=int, help='Number of nodes.', default=100)
parser.add_argument('--shuffle', type=str, help='Shuffle the dataset.', default=True)
parser.add_argument('--epochs', type=int, help='Epochs to train the model.', default=10)

parser.add_argument('--n_hidden', type=int, help='Number of hidden dimensions in the nn.', default=100)
parser.add_argument('--n_layers_enc', type=int, help='Number of layers in the encoder network.', default=1)
parser.add_argument('--n_layers_dec', type=int, help='Number of layers in the decoder network.', default=1)
parser.add_argument('--dropout', type=float, help='Dropout rate (1 - keep probability).', default=0.5)
parser.add_argument('--learning_rate', type=float, help='Learning rate of the optimization algorithm.', default=0.001)
parser.add_argument('--beta', default=5., type=float, help='beta is a hyperparameter in SDNE.')
parser.add_argument('--alpha', type=float, default=1e-2, help='alpha is a hyperparameter in SDNE.')
parser.add_argument('--theta', type=float, default=1, help='alpha is a hyperparameter in SDNE.')
parser.add_argument('--nu', type=float, default=1e-5, help='nu is a hyperparameter in SDNE.')
parser.add_argument('--gamma', type=float, default=1e2, help='gamma is a hyperparameter to multiply the add loss function.')
parser.add_argument('--early_stopping', type=str, default=False, help='Bool to specify if to use early stoping.')
parser.add_argument('--gradient_clipping', type=str, default=False, help='Bool to specify if to use gradient clipping.')
parser.add_argument('--stadardize_losses', type=str, default=False, help='Bool to specify if to standardize the value of loss functions.')
parser.add_argument('--eigen_loss_type', type=str, default="norm", help='Type of loss to compute the eigenvalues.')

if __name__ == '__main__':

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # parse args
    args = parser.parse_args()

    # convert to boolean
    args.sample = str_2_bool(args.sample)
    args.shuffle = str_2_bool(args.shuffle)
    args.early_stopping = str_2_bool(args.early_stopping)
    args.gradient_clipping = str_2_bool(args.gradient_clipping)
    args.stadardize_losses = str_2_bool(args.stadardize_losses)

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
    loss_eigen = LossEigen(loss_type=args.eigen_loss_type)

    # initialize tqdm
    pbar = tqdm(range(args.epochs), total=args.epochs, desc=f"Running {args.model_name} model")
    epochs_tot_loss, epochs_global_loss, epochs_local_loss, epochs_reg_loss, epochs_eigen_loss = [], [], [], [], []
    epochs_predictions = []

    # SDNE TRAINING: consists of computing gradients for each cov-batch, which contains all samples for a given covariance between graphs
    # SDNE TRAINING: accumulates gradients on the epoch level
    for epoch in pbar:

        opt1.zero_grad()
        opt2.zero_grad()

        epoch_results = []
        batch_tot_loss1, batch_global_loss1, batch_local_loss1, batch_reg_loss1, batch_eigen_loss1 = [], [], [], [], []
        batch_tot_loss2, batch_global_loss2, batch_local_loss2, batch_reg_loss2, batch_eigen_loss2 = [], [], [], [], []
        batch_predictions = []
        for cov in sim.covs:

            filtered_data_list = [data for data in dataset_list if (np.round(data.y.item(), 1) == cov)]
            filtered_loader = DataLoader(filtered_data_list, batch_size=args.batch_size, shuffle=args.shuffle)

            lt1_tot, lg1_tot, ll1_tot, lr1_tot, le1_tot = 0, 0, 0, 0, 0
            lt2_tot, lg2_tot, ll2_tot, lr2_tot, le2_tot = 0, 0, 0, 0, 0
            for data in filtered_loader:
                # Move data to the appropriate device
                data = data.to(device)

                # get inputs
                x1 = data.x[0, :, :].to(device)
                x2 = data.x[1, :, :].to(device)

                # create global loss parameter matrix
                b1_mat, b2_mat = torch.ones_like(x1), torch.ones_like(x2)
                b1_mat[x1 != 0], b2_mat[x2 != 0] = args.beta, args.beta

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

                ## compute total loss
                ## lg ~ ladd >>> lr > ll
                if args.stadardize_losses:
                    l1_sum = lg1.item() + ll1.item() + lr1.item() + le1.item()
                    lg1 /= l1_sum
                    ll1 /= l1_sum
                    lr1 /= l1_sum
                    le1 /= l1_sum
                lt1 = (args.alpha * lg1) + (args.theta * ll1) + (args.nu * lr1) + (args.gamma * le1)

                lt1_tot += lt1
                lg1_tot += lg1
                ll1_tot += ll1
                lr1_tot += lr1
                le1_tot += le1

                # compute loss functions II
                ll2 = loss_local.forward(adj=x2, z=z2)
                lg2 = loss_global.forward(adj=x2, x=x2_hat, b_mat=b2_mat)
                lr2 = loss_reg.forward(model=model2)
                le2 = loss_eigen.forward(adj=x2, x=x2_hat)

                ## compute total loss
                ## g ~ ladd >>> lr > ll
                if args.stadardize_losses:
                    l2_sum = lg2.item() + ll2.item() + lr2.item() + le2.item()
                    lg2 /= l2_sum
                    ll2 /= l2_sum
                    lr2 /= l2_sum
                    le2 /= l2_sum
                lt2 = (args.alpha * lg2) + (args.theta * ll2) + (args.nu * lr2) + (args.gamma * le2)

                lt2_tot += lt2
                lg2_tot += lg2
                ll2_tot += ll2
                lr2_tot += lr2
                le2_tot += le2

            ## backward pass
            lt1_tot.backward()
            opt1.step()

            ## backward pass
            lt2_tot.backward()
            opt2.step()

            ## gradient clipping
            if args.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)

            batch_tot_loss1.append(lt1_tot.detach().item())
            batch_global_loss1.append(lg1_tot.detach().item())
            batch_local_loss1.append(ll1_tot.detach().item())
            batch_reg_loss1.append(lr1_tot.detach().item())
            batch_eigen_loss1.append(le1_tot.detach().item())

            batch_tot_loss2.append(lt2_tot.detach().item())
            batch_global_loss2.append(lg2_tot.detach().item())
            batch_local_loss2.append(ll2_tot.detach().item())
            batch_reg_loss2.append(lr2_tot.detach().item())
            batch_eigen_loss2.append(le2_tot.detach().item())

        ## early stopping
        if args.early_stopping:
            if early_stopper.early_stop(lt1_tot) and early_stopper.early_stop(lt2_tot):             
                break

        epochs_predictions.append(torch.tensor(batch_predictions).to(device))

        epochs_tot_loss.append(torch.stack([torch.tensor(batch_tot_loss1), torch.tensor(batch_tot_loss2)], axis=1).to(device))
        epochs_global_loss.append(torch.stack([torch.tensor(batch_global_loss1), torch.tensor(batch_global_loss2)], axis=1).to(device))
        epochs_local_loss.append(torch.stack([torch.tensor(batch_local_loss1), torch.tensor(batch_local_loss2)], axis=1).to(device))
        epochs_reg_loss.append(torch.stack([torch.tensor(batch_reg_loss1), torch.tensor(batch_reg_loss2)], axis=1).to(device))
        epochs_eigen_loss.append(torch.stack([torch.tensor(batch_eigen_loss1), torch.tensor(batch_eigen_loss2)], axis=1).to(device))

    # pred list to tensor
    epochs_predictions = torch.stack(epochs_predictions)
    epochs_tot_loss = torch.stack(epochs_tot_loss)
    epochs_global_loss = torch.stack(epochs_global_loss)
    epochs_local_loss = torch.stack(epochs_local_loss)
    epochs_reg_loss = torch.stack(epochs_reg_loss)
    epochs_eigen_loss = torch.stack(epochs_eigen_loss)

    pbar = tqdm(range(len(sim.n_simulations)), total=len(sim.n_simulations), desc=f"Running {args.model_name} model on test data")
    test_results = []
    with torch.no_grad():
        for n in pbar:

            simulation_results = []
            for cov in sim.covs:

                filtered_data_list = [data for data in dataset_list if (data.n_simulations == n) and (np.round(data.y.item(), 1) == cov)]
                filtered_loader = DataLoader(filtered_data_list, batch_size=args.batch_size, shuffle=args.shuffle)

                embeddings = [] 
                for data in filtered_loader:
                    # Move data to the appropriate device
                    data = data.to(device)

                    # get inputs
                    x1 = data.x[0, :, :].to(device)
                    x2 = data.x[1, :, :].to(device)

                    # create global loss parameter matrix
                    b1_mat, b2_mat = torch.ones_like(x1), torch.ones_like(x2)
                    b1_mat[x1 != 0], b2_mat[x2 != 0] = args.beta, args.beta

                    # forward pass
                    x1_hat, z1, z1_norm = model1.forward(x1)
                    x2_hat, z2, z2_norm = model2.forward(x2)
                    embeddings.append(torch.stack((z1.flatten().detach(), z2.flatten().detach()), dim=1))

                embeddings = torch.concat(embeddings)

                pred_cov = model1.compute_spearman_rank_correlation(x=embeddings[:,0], y=embeddings[:,1])

                simulation_results.append([pred_cov, cov])
            
            simulation_results = torch.tensor(simulation_results).to(device)
            test_results.append(simulation_results)
                        
    test_results = torch.stack(test_results)

    outargs = {
        "args": args
    }

    predictions = {
        "train_predictions": epochs_predictions.cpu(),
        "test_predictions": test_results.cpu(),
    }

    training_info = {
        "train_loss": epochs_tot_loss.cpu(),
    }

    epochs_loss = {
        "epochs_global_loss": epochs_global_loss.cpu(),
        "epochs_local_loss": epochs_local_loss.cpu(),
        "epochs_reg_loss": epochs_reg_loss.cpu(),
        "epochs_eigen_loss": epochs_eigen_loss.cpu()
    }

    model_name = f"{args.model_name}_es" if args.early_stopping else args.model_name
    model_name = f"{model_name}_gc" if args.gradient_clipping else model_name
    model_name = f"{model_name}_sl" if args.stadardize_losses else model_name
    model_name = f"{model_name}_{args.eigen_loss_type}"

    if args.stadardize_losses:
        weights_name = f'alpha{int(args.alpha)}_theta{int(args.theta)}_nu{int(args.nu)}_gamma{int(args.gamma)}'
        model_name = f'{model_name}_{int(args.n_hidden)}_{int(args.n_layers_enc)}_{int(args.n_layers_dec)}_{int(args.epochs)}_{weights_name}'
    else:
        model_name = f'{model_name}_{int(args.n_hidden)}_{int(args.n_layers_enc)}_{int(args.n_layers_dec)}_{int(args.epochs)}'

    # check if file exists 
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{args.graph_name}/{model_name}"
    print(f"Saving data to {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save file
    if args.sample:
        save_pickle(path=f"{output_path}/sample_args.pkl", obj=outargs)
        save_pickle(path=f"{output_path}/sample_predictions.pkl", obj=predictions)
        save_pickle(path=f"{output_path}/sample_training_info.pkl", obj=training_info)
        save_pickle(path=f"{output_path}/sample_epochs_loss.pkl", obj=epochs_loss)
        torch.save(model1.state_dict(), f"{output_path}/model1_sample.pth")
        torch.save(model2.state_dict(), f"{output_path}/model2_sample.pth")
    else:
        save_pickle(path=f"{output_path}/args.pkl", obj=outargs)
        save_pickle(path=f"{output_path}/predictions.pkl", obj=predictions)
        save_pickle(path=f"{output_path}/training_info.pkl", obj=training_info)
        save_pickle(path=f"{output_path}/epochs_loss.pkl", obj=epochs_loss)
        torch.save(model1.state_dict(), f"{output_path}/model1.pth")
        torch.save(model2.state_dict(), f"{output_path}/model2.pth")