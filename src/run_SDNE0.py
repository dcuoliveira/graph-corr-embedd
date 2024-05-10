import torch
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from torch_geometric.data import DataLoader

from models.SDNE import SDNE
from data.Simulation1aLoader import Simulation1aLoader
from loss_functions.LossGlobal import LossGlobal
from loss_functions.LossLocal import LossLocal
from loss_functions.LossReg import LossReg
from loss_functions.LossAbsDistance import LossAbsDistance
from loss_functions.LossDistance import LossDistance

from utils.conn_data import save_pickle
from utils.parsers import str_2_bool

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="simulation1a")
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=False)
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1)
parser.add_argument('--model_name', type=str, help='Model name.', default="sdne0")
parser.add_argument('--n_nodes', type=int, help='Number of nodes.', default=100)
parser.add_argument('--shuffle', type=str, help='Shuffle the dataset.', default=True)

parser.add_argument('--n_hidden', type=int, help='Number of hidden dimensions in the nn.', default=100)
parser.add_argument('--n_layers_enc', type=int, help='Number of layers in the encoder network.', default=1)
parser.add_argument('--n_layers_dec', type=int, help='Number of layers in the decoder network.', default=1)
parser.add_argument('--dropout', type=float, help='Dropout rate (1 - keep probability).', default=0.5)
parser.add_argument('--learning_rate', type=float, help='Learning rate of the optimization algorithm.', default=0.001)
parser.add_argument('--epochs', type=int, help='Epochs to train the model.', default=10)
parser.add_argument('--beta', default=5., type=float, help='beta is a hyperparameter in SDNE.')
parser.add_argument('--alpha', type=float, default=1e-2, help='alpha is a hyperparameter in SDNE.')
parser.add_argument('--gamma', type=float, default=1e3, help='gamma is a hyperparameter to multiply the add loss function.')
parser.add_argument('--nu', type=float, default=1e-5, help='nu is a hyperparameter in SDNE.')

if __name__ == '__main__':

    # parse args
    args = parser.parse_args()

    # convert to boolean
    args.sample = str_2_bool(args.sample)
    args.shuffle = str_2_bool(args.shuffle)

    # define dataset
    sim = Simulation1aLoader(name=args.dataset_name, sample=args.sample)
    train_loader = sim.create_graph_loader(batch_size=args.batch_size)
    test_dataset_list = sim.create_graph_list()

    # define model
    model1 = SDNE(node_size=args.n_nodes,
                  n_hidden=args.n_hidden,
                  n_layers_enc=args.n_layers_enc,
                  n_layers_dec=args.n_layers_dec,
                  bias_enc=True,
                  bias_dec=True,
                  droput=args.dropout)
    
    model2 = SDNE(node_size=args.n_nodes,
                  n_hidden=args.n_hidden,
                  n_layers_enc=args.n_layers_enc,
                  n_layers_dec=args.n_layers_dec,
                  bias_enc=True,
                  bias_dec=True,
                  droput=args.dropout)
    
    # define optimizer
    opt1 = optim.Adam(model1.parameters(), lr=args.learning_rate)
    opt2 = optim.Adam(model2.parameters(), lr=args.learning_rate)

    # define loss functions
    loss_local = LossLocal()
    loss_global = LossGlobal()
    loss_reg = LossReg()
    loss_dis = LossDistance()
    loss_abs_dis = LossAbsDistance()

    # initialize tqdm
    pbar = tqdm(range(args.epochs))

    epochs_tot_loss, epochs_global_loss, epochs_local_loss, epochs_reg_loss = [], [], [], []
    epochs_predictions = []
    for epoch in pbar:

        batch_tot_loss1, batch_global_loss1, batch_local_loss1, batch_reg_loss1 = [], [], [], []
        batch_tot_loss2, batch_global_loss2, batch_local_loss2, batch_reg_loss2 = [], [], [], []
        batch_predictions = []
        for data in train_loader:

            opt1.zero_grad()
            opt2.zero_grad()
            
            # get inputs
            x1 = data.x[0, :, :]
            x2 = data.x[1, :, :]

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

        epochs_predictions.append(torch.tensor(batch_predictions))

        epochs_tot_loss.append(torch.stack([torch.tensor(batch_tot_loss1), torch.tensor(batch_tot_loss2)], axis=1))
        epochs_global_loss.append(torch.stack([torch.tensor(batch_global_loss1), torch.tensor(batch_global_loss2)], axis=1))
        epochs_local_loss.append(torch.stack([torch.tensor(batch_local_loss1), torch.tensor(batch_local_loss2)], axis=1))
        epochs_reg_loss.append(torch.stack([torch.tensor(batch_reg_loss1), torch.tensor(batch_reg_loss2)], axis=1))
        
        # update tqdm
        pbar.update(1)
        pbar.set_description("SDNE Train Epoch: %d, Train Loss I & II: %.4f & %.4f" % (epoch, batch_tot_loss1[-1].detach().item(), batch_tot_loss2[-1].detach().item()))

    # pred list to tensor
    epochs_predictions = torch.stack(epochs_predictions)
    epochs_tot_loss = torch.stack(epochs_tot_loss)
    epochs_global_loss = torch.stack(epochs_global_loss)
    epochs_local_loss = torch.stack(epochs_local_loss)
    epochs_reg_loss = torch.stack(epochs_reg_loss)

    pbar = tqdm(sim.n_simulations, total=len(sim.n_simulations), desc="Running SDNE model on Test Data")
    test_results = []
    with torch.no_grad():
        for n in pbar:

            simulation_results = []
            for cov in sim.covs:

                filtered_data_list = [data for data in test_dataset_list if (data.n_simulations == n) and (data.y.item() == cov)]
                filtered_loader = DataLoader(filtered_data_list, batch_size=args.batch_size, shuffle=args.shuffle)

                embeddings = [] 
                for data in filtered_loader:
                    # get inputs
                    x1 = data.x[0, :, :]
                    x2 = data.x[1, :, :]

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
            
            simulation_results = torch.tensor(simulation_results)
            test_results.append(simulation_results)
            
            pbar.update(1)
            pbar.set_description(f"Test Simulation: {n}")
            
    test_results = torch.stack(test_results)

    results = {
        "args": args,
        "train_predictions": epochs_predictions,
        "test_predictions": test_results,
        "train_loss": epochs_tot_loss,
        "epochs_global_loss": epochs_global_loss,
        "epochs_local_loss": epochs_local_loss,
        "epochs_reg_loss": epochs_reg_loss,
    }

    model_name = f'{args.model_name}_{int(args.n_hidden)}_{int(args.n_layers_enc)}_{int(args.n_layers_dec)}_{int(args.epochs)}'

    # check if file exists
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save file
    if args.sample:
        save_pickle(path=f"{output_path}/sample_results.pkl", obj=results)
        torch.save(model1.state_dict(), f"{output_path}/model1_sample.pth")
        torch.save(model2.state_dict(), f"{output_path}/model2_sample.pth")
    else:
        save_pickle(path=f"{output_path}/results.pkl", obj=results)
        torch.save(model1.state_dict(), f"{output_path}/model1.pth")
        torch.save(model2.state_dict(), f"{output_path}/model2.pth")