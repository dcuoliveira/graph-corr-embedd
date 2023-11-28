import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import argparse
import os
from tqdm import tqdm

from models.SDNE import SDNE
from loss_functions.LossGlobal import LossGlobal
from loss_functions.LossLocal import LossLocal
from loss_functions.LossReg import LossReg
from data.ExamplesLoader import ExamplesLoader
from data.DataLoad import Dataload

from utils.conn_data import save_pickle

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="cora")
parser.add_argument('--model_name', type=str, help='Model name.', default="sdne")
parser.add_argument('--n_hidden', type=int, help='Number of hidden dimensions in the nn.', default=100)
parser.add_argument('--n_layers_enc', type=int, help='Number of layers in the encoder network.', default=1)
parser.add_argument('--n_layers_dec', type=int, help='Number of layers in the decoder network.', default=1)
parser.add_argument('--dropout', type=float, help='Dropout rate (1 - keep probability).', default=0.5)
parser.add_argument('--learning_rate', type=float, help='Learning rate of the optimization algorithm.', default=0.001)
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=100)
parser.add_argument('--epochs', type=int, help='Epochs to train the model.', default=1000)
parser.add_argument('--beta', default=5., type=float, help='beta is a hyperparameter in SDNE.')
parser.add_argument('--alpha', type=float, default=1e-2, help='alpha is a hyperparameter in SDNE.')
parser.add_argument('--nu', type=float, default=1e-5, help='nu is a hyperparameter in SDNE.')

if __name__ == '__main__':

    args = parser.parse_args()

    # load graph data
    el =  ExamplesLoader(example_name=args.dataset_name)
    G, Adj, n_nodes = el.G, el.Adj, el.n_nodes

    # define model
    model = SDNE(node_size=n_nodes,
                 n_hidden=args.n_hidden,
                 n_layers_enc=args.n_layers_enc,
                 n_layers_dec=args.n_layers_dec,
                 bias_enc=True,
                 bias_dec=True,
                 droput=args.dropout)

    # define loss functions
    loss_local = LossLocal()
    loss_global = LossGlobal()
    loss_reg = LossReg()

    # define optimizer
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)

    # create data loader
    Data = Dataload(Adj, n_nodes)
    Data = DataLoader(Data, batch_size=args.batch_size, shuffle=True)

    # train model
    model.train()

    # initialize tqdm
    pbar = tqdm(total=args.epochs)

    xs_train, zs_train, z_norms_train, epochs_loss_train_tot = [], [], [], []
    for epoch in range(1, args.epochs + 1):

        loss_train_tot, loss_global_tot, loss_local_tot, loss_reg_tot = 0, 0, 0, 0
        for index in Data:

            # select batch info
            adj_batch = Adj[index]
            adj_mat = adj_batch[:, index]
            b_mat = torch.ones_like(adj_batch)
            b_mat[adj_batch != 0] = args.beta

            # set previous gradients to zero
            opt.zero_grad()

            # forward pass model
            x, z, z_norm = model(adj_batch)

            # save outputs
            xs_train.append(x.detach())
            zs_train.append(z.detach())
            z_norms_train.append(z_norm.detach())

            # compute loss functions
            ll = loss_local.forward(adj=adj_mat, z=z)
            lg = loss_global.forward(adj=adj_batch, x=x, b_mat=b_mat)
            lr = loss_reg.forward(model=model)

            # compute total loss
            # lg >>> lr > ll
            lt = (args.alpha * lg) + ll + (args.nu * lr)

            loss_train_tot += lt
            loss_global_tot += lg
            loss_local_tot += ll
            loss_reg_tot += lr

            # backward pass
            lt.backward()
            opt.step()

        # update tqdm
        pbar.update(1)
        pbar.set_description("Epoch: %d, Train Loss: %.4f" % (epoch, loss_train_tot))

        # save loss
        epochs_loss_train_tot.append(loss_train_tot.detach())
    pbar.close()

    # evaluate model
    model.eval()

    xs_eval, zs_eval, z_norms_eval = [], [], []
    with torch.no_grad():

        # forward pass model
        adj_batch = Adj
        adj_mat = adj_batch[:, :]
        b_mat = torch.ones_like(adj_batch)
        b_mat[adj_batch != 0] = args.beta
        x, z, z_norm = model(adj_batch)

        # save outputs
        xs_eval.append(x.detach())
        zs_eval.append(z.detach())
        z_norms_eval.append(z_norm.detach())

        # compute loss functions
        ll = loss_local.forward(adj=adj_mat, z=z)
        lg = loss_global.forward(adj=adj_batch, x=x, b_mat=b_mat)
        lr = loss_reg.forward(model=model)

        # compute total loss
        loss_eval_tot = (args.alpha * lg) + ll + (args.nu * lr)

    # save results
    results = {

        "xs_train": xs_train[-1],
        "zs_train": zs_train[-1],
        "z_norms_train": z_norms_train[-1],

        "epochs_train_loss": epochs_loss_train_tot,
        "model": model, 

        "xs_eval": xs_eval,
        "zs_eval": zs_eval,
        "z_norms_eval": z_norms_eval,

        "args": args

    }

    # check if dir exists
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "data", "outputs", args.dataset_name, args.model_name)):
        os.makedirs(os.path.join(os.path.dirname(__file__), "data", "outputs", args.dataset_name, args.model_name))

    pbar = tqdm(total=len(results))
    for key, value in results.items():
        save_pickle(obj={key: value},
                    path=os.path.join(os.path.dirname(__file__),
                                    "data",
                                    "outputs",
                                    args.dataset_name,
                                    args.model_name,
                                    f"{key}.pickle"))
        pbar.update(1)
        pbar.set_description(f"Saving {key}")

    pbar.close()
