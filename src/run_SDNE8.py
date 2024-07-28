import torch
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from torch_geometric.data import DataLoader

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
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=False)
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1, choices=[1])
parser.add_argument('--model_name', type=str, help='Model name.', default="sdne8")
parser.add_argument('--n_nodes', type=int, help='Number of nodes.', default=100)
parser.add_argument('--shuffle', type=str, help='Shuffle the dataset.', default=True)
parser.add_argument('--epochs', type=int, help='Epochs to train the model.', default=10)

parser.add_argument('--n_hidden', type=int, help='Number of hidden dimensions in the nn.', default=100)
parser.add_argument('--n_layers_enc', type=int, help='Number of layers in the encoder network.', default=1)
parser.add_argument('--n_layers_dec', type=int, help='Number of layers in the decoder network.', default=1)
parser.add_argument('--dropout', type=float, help='Dropout rate (1 - keep probability).', default=0.5)
parser.add_argument('--learning_rate', type=float, help='Learning rate of the optimization algorithm.', default=0.001)
parser.add_argument('--theta', default=5., type=float, help='beta is a hyperparameter in SDNE.')
parser.add_argument('--alpha', type=float, default=1, help='weight hyperparameter of the global loss.')
parser.add_argument('--beta', default=1, type=float, help='weight hyperparameter of the local loss.')
parser.add_argument('--gamma', type=float, default=1, help='weight hyperparameter of the regularization loss.')
parser.add_argument('--nu', type=float, default=1, help='weight hyperparameter of the eigen loss.')
parser.add_argument('--decay', type=float, default=0.9, help='Decay rate for moving averages.')

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
    loss_eigen = LossEigen()

    # initialize tqdm
    pbar = tqdm(range(args.epochs), total=len(sim.n_simulations), desc=f"Running {args.model_name} model")
    epochs_tot_loss, epochs_global_loss, epochs_local_loss, epochs_reg_loss, epochs_eigen_loss = [], [], [], [], []
    epochs_predictions = []

    # SDNE TRAINING: consists of computing gradients for each cov-batch, which contains all samples for a given covariance between graphs
    # SDNE TRAINING: accumulates gradients on the epoch level
    lg1_mavg, ll1_mavg, lr1_mavg, le1_mavg = 1, 1, 1, 1
    lg2_mavg, ll2_mavg, lr2_mavg, le2_mavg = 1, 1, 1, 1
    for epoch in pbar:

        opt1.zero_grad()
        opt2.zero_grad()

        epoch_results = []
        for cov in sim.covs:

            filtered_data_list = [data for data in dataset_list if (data.y.item() == cov)]
            filtered_loader = DataLoader(filtered_data_list, batch_size=args.batch_size, shuffle=args.shuffle)
        
            batch_tot_loss1, batch_global_loss1, batch_local_loss1, batch_reg_loss1, batch_eigen_loss1 = [], [], [], [], []
            batch_tot_loss2, batch_global_loss2, batch_local_loss2, batch_reg_loss2, batch_eigen_loss2 = [], [], [], [], []
            batch_predictions = []

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
                pred_cov = model1.compute_spearman_rank_correlation_tensor(x=z1.flatten().detach(), y=z2.flatten().detach())

                # store pred and true values
                batch_predictions.append([pred_cov, data.y])

                # compute loss functions I
                ll1 = loss_local.forward(adj=x1, z=z1)
                lg1 = loss_global.forward(adj=x1, x=x1_hat, b_mat=b1_mat)
                lr1 = loss_reg.forward(model=model1)
                le1 = loss_eigen.forward(adj=x1, x=x1_hat)

                # Update moving averages
                lg1_mavg = args.decay * lg1_mavg + (1 - args.decay) * lg1.item()
                ll1_mavg = args.decay * ll1_mavg + (1 - args.decay) * ll1.item()
                lr1_mavg = args.decay * lr1_mavg + (1 - args.decay) * lr1.item()
                le1_mavg = args.decay * le1_mavg + (1 - args.decay) * le1.item()

                # Standardize losses
                standard_lg1 = lg1 / lg1_mavg
                standard_ll1 = ll1 / ll1_mavg
                standard_lr1 = lr1 / lr1_mavg
                standard_le1 = le1 / le1_mavg

                ## compute total loss
                ## lg ~ ladd >>> lr > ll
                lt1 = (args.alpha * standard_lg1) + (args.beta * standard_ll1) + (args.gamma * standard_lr1) + (args.nu * standard_le1)

                lt1_tot += lt1
                lg1_tot += standard_lg1
                ll1_tot += standard_ll1
                lr1_tot += standard_lr1
                le1_tot += standard_le1

                # compute loss functions II
                ll2 = loss_local.forward(adj=x2, z=z2)
                lg2 = loss_global.forward(adj=x2, x=x2_hat, b_mat=b2_mat)
                lr2 = loss_reg.forward(model=model2)
                le2 = loss_eigen.forward(adj=x2, x=x2_hat)

                # Update moving averages
                lg2_mavg = args.decay * lg2_mavg + (1 - args.decay) * lg2.item()
                ll2_mavg = args.decay * ll2_mavg + (1 - args.decay) * ll2.item()
                lr2_mavg = args.decay * lr2_mavg + (1 - args.decay) * lr2.item()
                le2_mavg = args.decay * le2_mavg + (1 - args.decay) * le2.item()

                # Standardize losses
                standard_lg2 = lg2 / lg2_mavg
                standard_ll2 = ll2 / ll2_mavg
                standard_lr2 = lr2 / lr2_mavg
                standard_le2 = le2 / le2_mavg

                ## compute total loss
                ## g ~ ladd >>> lr > ll
                lt2 = (args.alpha * standard_lg2) + (args.beta * standard_ll2) + (args.gamma * standard_lr2) + (args.nu * standard_le2)

                lt2_tot += lt2
                lg2_tot += standard_lg2
                ll2_tot += standard_ll2
                lr2_tot += standard_lr2
                le2_tot += standard_le2

            ## backward pass
            lt1_tot.backward()
            opt1.step()

            ## backward pass
            lt2_tot.backward()
            opt2.step()

            if early_stopper.early_stop(lt1_tot) and early_stopper.early_stop(lt2_tot):             
                break

            ## gradient clipping
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

        epochs_predictions.append(torch.tensor(batch_predictions).to(device))

        epochs_tot_loss.append(torch.stack([torch.tensor(batch_tot_loss1), torch.tensor(batch_tot_loss2)], axis=1).to(device))
        epochs_global_loss.append(torch.stack([torch.tensor(batch_global_loss1), torch.tensor(batch_global_loss2)], axis=1).to(device))
        epochs_local_loss.append(torch.stack([torch.tensor(batch_local_loss1), torch.tensor(batch_local_loss2)], axis=1).to(device))
        epochs_reg_loss.append(torch.stack([torch.tensor(batch_reg_loss1), torch.tensor(batch_reg_loss2)], axis=1).to(device))
        epochs_eigen_loss.append(torch.stack([torch.tensor(batch_eigen_loss1), torch.tensor(batch_eigen_loss2)], axis=1).to(device))
        
        # update tqdm
        pbar.update(1)

    # pred list to tensor
    epochs_predictions = torch.stack(epochs_predictions)
    epochs_tot_loss = torch.stack(epochs_tot_loss)
    epochs_global_loss = torch.stack(epochs_global_loss)
    epochs_local_loss = torch.stack(epochs_local_loss)
    epochs_reg_loss = torch.stack(epochs_reg_loss)
    epochs_eigen_loss = torch.stack(epochs_eigen_loss)

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

                pred_cov = model1.compute_spearman_rank_correlation_tensor(x=z1.flatten().detach(), y=z2.flatten().detach())

                simulation_results.append([pred_cov, cov])
            
            simulation_results = torch.tensor(simulation_results).to(device)
            test_results.append(simulation_results)
                        
    test_results = torch.stack(test_results)

    outargs = {
        "args": args
    }

    predictions = {
        "train_predictions": epochs_predictions,
        "test_predictions": test_results,
    }

    training_info = {
        "train_loss": epochs_tot_loss,
    }

    epochs_loss = {
        "epochs_global_loss": epochs_global_loss,
        "epochs_local_loss": epochs_local_loss,
        "epochs_reg_loss": epochs_reg_loss,
        "epochs_eigen_loss": epochs_eigen_loss
    }

    weights_name = f'alpha{int(args.alpha)}_beta{int(args.beta)}_gamma{int(args.gamma)}_nu{int(args.nu)}'
    model_name = f'{args.model_name}_{int(args.n_hidden)}_{int(args.n_layers_enc)}_{int(args.n_layers_dec)}_{int(args.epochs)}_{weights_name}'

    # check if file exists 
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{args.graph_name}/{model_name}"
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
