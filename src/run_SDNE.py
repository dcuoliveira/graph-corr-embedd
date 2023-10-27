import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import numpy as np
import argparse

from models.SDNE import SDNE
from loss_functions.LossGlobal import LossGlobal
from loss_functions.LossLocal import LossLocal
from loss_functions.LossReg import LossReg
from data.ExamplesLoader import ExamplesLoader
from data.DataLoad import Dataload

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--n_hidden', type=int, help='Number of hidden dimensions in the nn.', default=1)
parser.add_argument('--dropout', type=float, help='Dropout rate (1 - keep probability).', default=0.5)
parser.add_argument('--learning_rate', type=float, help='Learning rate of the optimization algorithm.', default=0.001)
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=100)
parser.add_argument('--epochs', type=int, help='Epochs to train the model.', default=1000)
parser.add_argument('--beta', default=5., type=float, help='beta is a hyperparameter in SDNE.')
parser.add_argument('--alpha', type=float, default=1e-2, help='alpha is a hyperparameter in SDNE.')
parser.add_argument('--nu1', type=float, default=1e-5, help='nu1 is a hyperparameter in SDNE.')

if __name__ == '__main__':

    args = parser.parse_args()

    # load graph data
    el =  ExamplesLoader(example_name="cora")
    G, Adj, Node = el.G, el.Adj, el.Node

    # define model
    model = SDNE(node_size=Node, n_hidden=args.n_hidden, droput=args.dropout)

    # define loss functions
    loss_global = LossGlobal()
    loss_local = LossLocal()
    loss_reg = LossReg()

    # define optimizer
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)

    # create data loader
    Data = Dataload(Adj, Node)
    Data = DataLoader(Data, batch_size=args.batch_size, shuffle=True)

    # train model
    model.train()

    for epoch in range(1, args.epochs + 1):

        loss_tot_tot, loss_global_tot, loss_local_tot, loss_reg_tot = 0, 0, 0, 0
        for index in Data:

            opt.zero_grad()

            adj_batch = Adj[index]
            adj_mat = adj_batch[:, index]
            b_mat = torch.ones_like(adj_batch)
            b_mat[adj_batch != 0] = args.beta

            x, z, z_norm = model(adj_batch, adj_mat, b_mat)

            loss_global = loss_global.forward(adj=adj_batch, x=x, b_mat=b_mat)
            loss_local = loss_global.forward(adj=adj_batch, x=x, b_mat=b_mat)
            loss_reg = loss_global.forward(adj=adj_batch, x=x, b_mat=b_mat)

            loss_tot = (args.alpha * loss_global) + loss_local + (args.nu * loss_reg)
            loss_tot.backward()
            opt.step()

            loss_tot_tot += loss_tot
            loss_global_tot += loss_global
            loss_local_tot += loss_local
            loss_reg_tot += loss_reg

    model.eval()
    embedding = model.savector(Adj)
    outVec = embedding.detach().numpy()
    np.savetxt(args.output, outVec)

