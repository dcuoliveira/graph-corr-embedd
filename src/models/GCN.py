import torch.nn as nn
import torch
import dgl
from dgl.nn.pytorch.conv import GraphConv, GINConv, GATConv

from models.MLP import MLP
from src.model_utils.ApplyNodeFunc import ApplyNodeFunc

class GCN(nn.Module):

    def __init__(self, input_dim, type='gcn', n_hidden=16, n_layers=3, dropout=0.1, **kwargs):
        super(GCN, self).__init__()
        self.type = type
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.input_dim = input_dim

        if type == 'gcn':
            self.layer0 = GraphConv(input_dim, n_hidden)
            for i in range(self.n_layers-1):
                self.add_module('layer{}'.format(i + 1), GraphConv(n_hidden, n_hidden))

        elif type == 'gin':
            # removed batch normalisation + linear function for graph pooling + stacking all representations
            #self.nmlp_layers = **num_mlp_layers
            for layer in range(self.n_layers):
                if layer == 0:
                    mlp = MLP(2, input_dim, n_hidden, n_hidden, dropout)
                else:
                    mlp = MLP(2, n_hidden, n_hidden, n_hidden, dropout)

                self.add_module('layer{}'.format(layer), GINConv(ApplyNodeFunc(mlp), 'sum', 0, learn_eps=True))
                #self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        elif type == 'gat':
            self.layer0 = GATConv(input_dim, n_hidden, num_heads=1, residual=False)
            for i in range(n_layers - 1):
                self.add_module('layer{}'.format(i + 1),
                                GATConv(n_hidden, n_hidden, num_heads=1, residual=False))

    def forward(self, graph):

        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        x = graph.ndata['node_attr']

        if self.type == 'identity':
            return x

        for i in range(self.n_layers-1):
            x = torch.relu(self._modules['layer{}'.format(i)](graph,x).squeeze())
            x = self.dropout(x)
        x = self._modules['layer{}'.format(self.n_layers-1)](graph,x).squeeze()

        return x


