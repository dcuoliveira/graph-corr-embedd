import dgl
import numpy as np

def split_list_of_datasets(datasets: list, train_ratio: float = 0.8, val_ratio: float = 0.1):
    n = len(datasets)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    
    train_idx = np.random.random_integers(0, n, size=train_size)
    train_set = set(train_idx)
    all_indices = set(range(n))
    available_indices = all_indices - train_set
    available_indices = list(available_indices)
    val_idx = np.random.choice(available_indices, size=val_size, replace=False)
    available_indices = all_indices - set(val_idx) - train_set
    test_idx = np.array(list(available_indices))
    
    train_list = [datasets[i] for i in train_idx]
    val_list = [datasets[i] for i in val_idx]
    test_list = [datasets[i] for i in test_idx]

    return train_list, val_list, test_list

def tensor_to_dgl_graph_with_features(adj, node_features):
    # Get the source and destination nodes from the adjacency matrix
    src, dst = adj.nonzero(as_tuple=True)
    # Create a DGLGraph
    g = dgl.graph((src, dst))
    # Assign node features
    g.ndata['node_attr'] = node_features
    return g


def tensor_to_dgl_graph(adj):
    # Get the source and destination nodes from the adjacency matrix
    src, dst = adj.nonzero(as_tuple=True)
    # Create a DGLGraph
    return dgl.graph((src, dst))

def str_2_list(val):
    return [int(x) for x in val.split(",")]

def str_2_bool(val):

    val = str(val)

    if val.lower() == "false":
        return False
    elif val.lower() == "true": 
        return True
    else:
        raise Exception("Invalid boolean value: {}".format(val))