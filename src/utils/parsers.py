import dgl

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