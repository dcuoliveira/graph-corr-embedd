import torch
import torch.optim as optim
import argparse
import os
from tqdm import tqdm

from models.SDNE import SDNE
from data.Simulation1Loader import Simulation1Loader
from loss_functions.LossGlobal import LossGlobal
from loss_functions.LossLocal import LossLocal
from loss_functions.LossReg import LossReg

from utils.conn_data import save_pickle

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="simulation1")
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=True)
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=1)
parser.add_argument('--model_name', type=str, help='Model name.', default="sdne")
parser.add_argument('--n_nodes', type=int, help='Number of nodes.', default=100)

parser.add_argument('--n_hidden', type=int, help='Number of hidden dimensions in the nn.', default=100)
parser.add_argument('--n_layers_enc', type=int, help='Number of layers in the encoder network.', default=1)
parser.add_argument('--n_layers_dec', type=int, help='Number of layers in the decoder network.', default=1)
parser.add_argument('--dropout', type=float, help='Dropout rate (1 - keep probability).', default=0.5)
parser.add_argument('--learning_rate', type=float, help='Learning rate of the optimization algorithm.', default=0.001)
parser.add_argument('--epochs', type=int, help='Epochs to train the model.', default=10)
parser.add_argument('--beta', default=5., type=float, help='beta is a hyperparameter in SDNE.')
parser.add_argument('--alpha', type=float, default=1e-2, help='alpha is a hyperparameter in SDNE.')
parser.add_argument('--nu', type=float, default=1e-5, help='nu is a hyperparameter in SDNE.')

if __name__ == '__main__':

    args = parser.parse_args()

    # define dataset
    sim = Simulation1Loader(name=args.dataset_name, sample=args.sample)
    loader = sim.create_graph_loader(batch_size=args.batch_size)

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

    # initialize tqdm
    pbar = tqdm(range(args.epochs))

    xs_train, zs_train, z_norms_train = [], [], []
    epochs_loss_train_tot, epochs_loss_global_tot, epochs_loss_local_tot, epochs_loss_reg_tot = [], [], [], []
    for epoch in pbar:

        loss_train_tot1, loss_global_tot1, loss_local_tot1, loss_reg_tot1 = 0, 0, 0, 0
        loss_train_tot2, loss_global_tot2, loss_local_tot2, loss_reg_tot2 = 0, 0, 0, 0
        for data in loader:
            # get inputs
            x1 = data.x[0, :, :]
            x2 = data.x[1, :, :]

            # create global loss parameter matrix
            b1_mat, b2_mat = torch.ones_like(x1), torch.ones_like(x2)
            b1_mat[x1 != 0], b2_mat[x2 != 0] = args.beta, args.beta

            # forward pass
            x1_hat, z1, z1_norm = model1.forward(x1)
            x2_hat, z2, z2_norm = model2.forward(x2)

            # compute covariance between embeddings (true target)
            cov = model1.compute_spearman_correlation(x=z1.flatten().detach(), y=z2.flatten().detach())

            # compute loss functions I
            ll1 = loss_local.forward(adj=x1, z=z1)
            lg1 = loss_global.forward(adj=x1, x=x1_hat, b_mat=b1_mat)
            lr1 = loss_reg.forward(model=model1)

            ## compute total loss
            ## lg >>> lr > ll
            lt1 = (args.alpha * lg1) + ll1 + (args.nu * lr1)

            loss_train_tot1 += lt1
            loss_global_tot1 += lg1
            loss_local_tot1 += ll1
            loss_reg_tot1 += lr1

            ## backward pass
            lt1.backward()
            opt1.step()

            # compute loss functions II
            ll2 = loss_local.forward(adj=x2, z=z2)
            lg2 = loss_global.forward(adj=x2, x=x2_hat, b_mat=b2_mat)
            lr2 = loss_reg.forward(model=model2)

            ## compute total loss
            ## lg >>> lr > ll
            lt2 = (args.alpha * lg2) + ll2 + (args.nu * lr2)

            loss_train_tot2 += lt2
            loss_global_tot2 += lg2
            loss_local_tot2 += ll2
            loss_reg_tot2 += lr2

            ## backward pass
            lt2.backward()
            opt2.step()

        # update tqdm
        pbar.update(1)
        pbar.set_description("Train Epoch: %d, Train Loss I & II: %.4f & %.4f" % (epoch, loss_train_tot1, loss_train_tot2))

        # save loss
        epochs_loss_train_tot.append([loss_train_tot1.detach(), loss_train_tot2.detach()])
        epochs_loss_global_tot.append([loss_global_tot1.detach(), loss_train_tot2.detach()])
        epochs_loss_local_tot.append([loss_local_tot1.detach(), loss_train_tot2.detach()])
        epochs_loss_reg_tot.append([loss_reg_tot1.detach(), loss_train_tot2.detach()])

    pbar = tqdm(enumerate(loader), total=len(loader))
    pred = []
    true = []
    with torch.no_grad():
        for s, data in pbar:
            # get inputs
            x1 = data.x[0, :, :]
            x2 = data.x[1, :, :]

            # create global loss parameter matrix
            b1_mat, b2_mat = torch.ones_like(x1), torch.ones_like(x2)
            b1_mat[x1 != 0], b2_mat[x2 != 0] = args.beta, args.beta

            # forward pass
            x1_hat, z1, z1_norm = model1.forward(x1)
            x2_hat, z2, z2_norm = model2.forward(x2)

            # compute covariance between embeddings (true target)
            cov = model1.compute_spearman_correlation(x=z1, y=z2)

            # store results
            pred.append(cov)
            true.append(data.y)

            # update tqdm
            pbar.update(1)
            pbar.set_description(f"Test Sample: {s}")
        
    # pred list to tensor
    pred = torch.tensor(pred)
    true = torch.tensor(true)

    results = {
        "pred": pred,
        "true": true,
        "train_total_loss": epochs_loss_train_tot,
        "train_local_loss": epochs_loss_local_tot,
        "train_global_loss": epochs_loss_global_tot,
        "train_reg_loss": epochs_loss_reg_tot,
    }

    # check if file exists
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{args.model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save file
    if args.sample:
        save_pickle(path=f"{output_path}/sample_results.pkl", obj=results)
    else:
        save_pickle(path=f"{output_path}/results.pkl", obj=results)

