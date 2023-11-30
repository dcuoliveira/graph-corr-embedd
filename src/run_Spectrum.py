import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import argparse
import os

from models.Spectrum import Spectrum
from data.ExamplesLoader import ExamplesLoader
from data.DataLoad import Dataload

from utils.conn_data import save_pickle

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--model_name', type=str, help='Model name.', default="spectrum")
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=100)
parser.add_argument('--epochs', type=int, help='Epochs to train the model.', default=1000)

if __name__ == '__main__':

    args = parser.parse_args()

    # load graph data
    el =  ExamplesLoader(example_name="cora")
    G, Adj, n_nodes = el.G, el.Adj, el.n_nodes

    # define model
    model = Spectrum()

    # # define optimizer
    # opt = optim.Adam(model.parameters(), lr=args.learning_rate)

    # create data loader
    Data = Dataload(Adj, n_nodes)
    Data = DataLoader(Data, batch_size=args.batch_size, shuffle=True)

    # train model
    xs_train, zs_train, z_norms_train = [], [], []
    for epoch in range(1, args.epochs + 1):

        loss_tot_tot, loss_global_tot, loss_local_tot, loss_reg_tot = 0, 0, 0, 0
        for index in Data:

            # select batch info
            adj_batch = Adj[index]
            # adj_mat = adj_batch[:, index]
            # b_mat = torch.ones_like(adj_batch)
            # b_mat[adj_batch != 0] = args.beta

            # # set previous gradients to zero
            # opt.zero_grad()

            # forward pass model
            z, v = model(adj_batch)

            # # save outputs
            # xs_train.append(x.detach())
            zs_train.append(z.detach())
            # z_norms_train.append(z_norm.detach())

            # # compute loss functions
            # lg = loss_global.forward(adj=adj_batch, x=x, b_mat=b_mat)
            # ll = loss_local.forward(adj=adj_mat, z=z)
            # lr = loss_reg.forward(model=model)

            # # compute total loss
            # lt = (args.alpha * lg) + ll + (args.nu * lr)

            # loss_tot_tot += lt
            # loss_global_tot += lg
            # loss_local_tot += ll
            # loss_reg_tot += lr

            # # backward pass
            # lt.backward()
            # opt.step()

    # # evaluate model
    # model.eval()

    # xs_eval, zs_eval, z_norms_eval = [], [], []
    # with torch.no_grad():

    #     # forward pass model
    #     adj_batch = Adj
    #     adj_mat = adj_batch[:, :]
    #     b_mat = torch.ones_like(adj_batch)
    #     b_mat[adj_batch != 0] = args.beta
    #     x, z, z_norm = model(Adj, Adj[:,:], b_mat)

    #     # save outputs
    #     xs_eval.append(x.detach())
    #     zs_eval.append(z.detach())
    #     z_norms_eval.append(z_norm.detach())

    #     # compute loss functions
    #     loss_global = loss_global.forward(adj=adj_batch, x=x, b_mat=b_mat)
    #     loss_local = loss_global.forward(adj=adj_batch, x=x, b_mat=b_mat)
    #     loss_reg = loss_global.forward(adj=adj_batch, x=x, b_mat=b_mat)

    #     # compute total loss
    #     eval_loss_tot = (args.alpha * loss_global) + loss_local + (args.nu * loss_reg)

    # save results
    results = {

        "xs_train": xs_train,
        "zs_train": zs_train,
        "z_norms_train": z_norms_train,

        "xs_eval": xs_eval,
        "zs_eval": zs_eval,
        "z_norms_eval": z_norms_eval,

        "loss_tot_tot": loss_tot_tot,
        "loss_global_tot": loss_global_tot,
        "loss_local_tot": loss_local_tot,
        "loss_reg_tot": loss_reg_tot,

        "eval_loss_tot": eval_loss_tot,
        "loss_global": loss_global,
        "loss_local": loss_local,
        "loss_reg": loss_reg,

        "args": args

    }

    # check if dir exists
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "data", "outputs", args.model_name)):
        os.makedirs(os.path.join(os.path.dirname(__file__), "data", "outputs", args.model_name))

    # save results
    save_pickle(obj=results,
                path=os.path.join(os.path.dirname(__file__),
                                  "data",
                                  "outputs",
                                  args.model_name,
                                  "results.pkl"))

    
