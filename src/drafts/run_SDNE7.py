import torch
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from torch_geometric.data import DataLoader
import random
import numpy as np

from models.SDNE import SDNE
from model_utils.EarlyStopper import EarlyStopper
from data.Simulation1aLoader import Simulation1aLoader
from data.Simulation1cLoader import Simulation1cLoader
from loss_functions.LossGlobal import LossGlobal
from loss_functions.LossLocal import LossLocal
from loss_functions.LossReg import LossReg

from utils.conn_data import save_pickle
from utils.parsers import str_2_bool

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="simulation1a")
parser.add_argument('--graph_name', type=str, help='Graph name.', default="erdos_renyi")
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=False)
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1, choices=[1])
parser.add_argument('--model_name', type=str, help='Model name.', default="sdne3")
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
parser.add_argument('--gamma', type=float, default=1e3, help='gamma is a hyperparameter to multiply the add loss function.')
parser.add_argument('--nu', type=float, default=1e-5, help='nu is a hyperparameter in SDNE.')

if __name__ == '__main__':

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # parse args
    args = parser.parse_args()

    # convert to boolean
    args.sample = str_2_bool(args.sample)
    args.shuffle = str_2_bool(args.shuffle)

    # define dataset
    if args.dataset_name == "simulation1a":
        sim = Simulation1aLoader(name=args.dataset_name, sample=args.sample)
    elif args.dataset_name == "simulation1c":
        sim = Simulation1cLoader(name=args.dataset_name, sample=args.sample, graph_name = args.graph_name)
    else:
        raise Exception('Dataset not found!')
    dataset_list = sim.create_graph_list()

    # train, validation, test split
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
    
    # define early stopper
    early_stopper = EarlyStopper(patience=10, min_delta=100)
    
    # define optimizer
    opt1 = optim.Adam(model1.parameters(), lr=args.learning_rate)
    opt2 = optim.Adam(model2.parameters(), lr=args.learning_rate)

    # define loss functions
    loss_local = LossLocal()
    loss_global = LossGlobal()
    loss_reg = LossReg()

    # initialize tqdm
    pbar = tqdm(range(args.epochs + 1), total=len(sim.n_simulations), desc=f"Running {args.model_name} model")
    epochs_tot_loss, epochs_global_loss, epochs_local_loss, epochs_reg_loss = [], [], [], []
    val_tot_loss, val_global_loss, val_local_loss, val_reg_loss = [], [], [], []
    epochs_predictions = []
    all_val_predictions = []

    # SDNE TRAINING: consists of computing gradients for each individual pairs of graphs
    # SDNE TRAINING: no accumulation of gradients
    for epoch in pbar:

        batch_tot_loss1, batch_global_loss1, batch_local_loss1, batch_reg_loss1 = [], [], [], []
        batch_tot_loss2, batch_global_loss2, batch_local_loss2, batch_reg_loss2 = [], [], [], []
        batch_predictions = []
        for data in train_list:

            opt1.zero_grad()
            opt2.zero_grad()
            
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

            ## compute total loss
            ## lg ~ ladd >>> lr > ll
            lt1 = (args.alpha * lg1) + ll1 + (args.nu * lr1)

            batch_tot_loss1.append(lt1)
            batch_global_loss1.append(lg1)
            batch_local_loss1.append(ll1)
            batch_reg_loss1.append(lr1)

            # compute loss functions II
            ll2 = loss_local.forward(adj=x2, z=z2)
            lg2 = loss_global.forward(adj=x2, x=x2_hat, b_mat=b2_mat)
            lr2 = loss_reg.forward(model=model2)

            ## compute total loss
            ## g ~ ladd >>> lr > ll
            lt2 = (args.alpha * lg2) + ll2 + (args.nu * lr2)

            batch_tot_loss2.append(lt2)
            batch_global_loss2.append(lg2)
            batch_local_loss2.append(ll2)
            batch_reg_loss2.append(lr2)

            ## backward pass
            lt1.backward()
            opt1.step()
                
            ## backward pass
            lt2.backward()
            opt2.step()

            ## gradient clipping
            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)

        val_predictions = []
        val_tot_loss1, val_global_loss1, val_local_loss1, val_reg_loss1 = [], [], [], []
        val_tot_loss2, val_global_loss2, val_local_loss2, val_reg_loss2 = [], [], [], []
        lt1_val_tot, lt2_val_tot = 0, 0
        with torch.no_grad():
            for data in val_list:

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
                val_predictions.append([pred_cov, data.y])

                # compute loss functions I
                ll1 = loss_local.forward(adj=x1, z=z1)
                lg1 = loss_global.forward(adj=x1, x=x1_hat, b_mat=b1_mat)
                lr1 = loss_reg.forward(model=model1)

                ## compute total loss
                ## lg ~ ladd >>> lr > ll
                lt1 = (args.alpha * lg1) + ll1 + (args.nu * lr1)
                lt1_val_tot += lt1

                val_tot_loss1.append(lt1)
                val_global_loss1.append(lg1)
                val_local_loss1.append(ll1)
                val_reg_loss1.append(lr1)

                # compute loss functions II
                ll2 = loss_local.forward(adj=x2, z=z2)
                lg2 = loss_global.forward(adj=x2, x=x2_hat, b_mat=b2_mat)
                lr2 = loss_reg.forward(model=model2)

                ## compute total loss
                ## g ~ ladd >>> lr > ll
                lt2 = (args.alpha * lg2) + ll2 + (args.nu * lr2)
                lt2_val_tot += lt2

                val_tot_loss2.append(lt2)
                val_global_loss2.append(lg2)
                val_local_loss2.append(ll2)
                val_reg_loss2.append(lr2)

        if early_stopper.early_stop(lt1_val_tot) and early_stopper.early_stop(lt2_val_tot):             
            break

        epochs_predictions.append(torch.tensor(batch_predictions).to(device))        
        epochs_tot_loss.append(torch.stack([torch.tensor(batch_tot_loss1), torch.tensor(batch_tot_loss2)], axis=1).to(device))
        epochs_global_loss.append(torch.stack([torch.tensor(batch_global_loss1), torch.tensor(batch_global_loss2)], axis=1).to(device))
        epochs_local_loss.append(torch.stack([torch.tensor(batch_local_loss1), torch.tensor(batch_local_loss2)], axis=1).to(device))
        epochs_reg_loss.append(torch.stack([torch.tensor(batch_reg_loss1), torch.tensor(batch_reg_loss2)], axis=1).to(device))

        all_val_predictions.append(torch.tensor(val_predictions).to(device))
        val_tot_loss.append(torch.stack([torch.tensor(val_tot_loss1), torch.tensor(val_tot_loss2)], axis=1).to(device))
        val_global_loss.append(torch.stack([torch.tensor(val_global_loss1), torch.tensor(val_global_loss2)], axis=1).to(device))
        val_local_loss.append(torch.stack([torch.tensor(val_local_loss1), torch.tensor(val_local_loss2)], axis=1).to(device))
        val_reg_loss.append(torch.stack([torch.tensor(val_reg_loss1), torch.tensor(val_reg_loss2)], axis=1).to(device))

        # update tqdm
        pbar.update(1)

    # pred list to tensor
    epochs_predictions = torch.stack(epochs_predictions)
    epochs_tot_loss = torch.stack(epochs_tot_loss)
    epochs_global_loss = torch.stack(epochs_global_loss)
    epochs_local_loss = torch.stack(epochs_local_loss)
    epochs_reg_loss = torch.stack(epochs_reg_loss)
    all_val_predictions = torch.stack(all_val_predictions)
    val_tot_loss = torch.stack(val_tot_loss)
    val_global_loss = torch.stack(val_global_loss)
    val_local_loss = torch.stack(val_local_loss)
    val_reg_loss = torch.stack(val_reg_loss)

    pbar = tqdm(sim.n_simulations, total=len(sim.n_simulations), desc=f"Running {args.model_name} model on test data")
    test_results = []
    with torch.no_grad():
        for n in pbar:

            simulation_results = []
            for cov in sim.covs:

                filtered_data_list = [data for data in dataset_list if (data.n_simulations == n) and (data.y.item() == cov)]
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
        "epochs_train_predictions": epochs_predictions,
        "epochs_val_predictions": all_val_predictions,
        "test_predictions": test_results,
    }

    training_info = {
        "train_loss": epochs_tot_loss,
        "epochs_global_loss": epochs_global_loss,
        "epochs_local_loss": epochs_local_loss,
        "epochs_reg_loss": epochs_reg_loss,
        "val_loss": val_tot_loss,
        "val_global_loss": val_global_loss,
        "val_local_loss": val_local_loss,
        "val_reg_loss": val_reg_loss,
    }

    model_name = f'{args.model_name}_{int(args.n_hidden)}_{int(args.n_layers_enc)}_{int(args.n_layers_dec)}_{int(args.epochs)}'

    # check if file exists 
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{args.graph_name}/{model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save file
    if args.sample:
        save_pickle(path=f"{output_path}/sample_args.pkl", obj=outargs)
        save_pickle(path=f"{output_path}/sample_predictions.pkl", obj=predictions)
        save_pickle(path=f"{output_path}/sample_training_info.pkl", obj=training_info)
        torch.save(model1.state_dict(), f"{output_path}/model1_sample.pth")
        torch.save(model2.state_dict(), f"{output_path}/model2_sample.pth")
    else:
        save_pickle(path=f"{output_path}/args.pkl", obj=outargs)
        save_pickle(path=f"{output_path}/predictions.pkl", obj=predictions)
        save_pickle(path=f"{output_path}/training_info.pkl", obj=training_info)
        torch.save(model1.state_dict(), f"{output_path}/model1.pth")
        torch.save(model2.state_dict(), f"{output_path}/model2.pth")
