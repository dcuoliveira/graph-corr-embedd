import torch
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from torch_geometric.data import DataLoader

from node_encodings.NodeEncodingsParser import NodeEncodingsParser
from models.GCN import GCN
from models.SGNN import SGNN
from data.Simulation1aLoader import Simulation1aLoader

from utils.conn_data import save_pickle
from utils.parsers import str_2_bool

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--dataset_name', type=str, help='Dataset name.', default="simulation1a")
parser.add_argument('--sample', type=str, help='Boolean if sample graph to save.', default=True)
parser.add_argument('--batch_size', type=int, help='Batch size to traint the model.', default=2)
parser.add_argument('--model_name', type=str, help='Model name.', default="sdne2")
parser.add_argument('--shuffle', type=str, help='Shuffle the dataset.', default=True)
parser.add_argument('--epochs', type=int, help='Epochs to train the model.', default=10)

parser.add_argument('--encoding_type', type=str, help='Type of encoding to use.', default='degree')
parser.add_argument('--n_nodes', type=int, help='Number of nodes in the graph.', default=10)
parser.add_argument('--n_hidden_encoder', type=int, help='Number of hidden units in the model.', default=15)
parser.add_argument('--similarity', type=str, help='Similarity metric to use.', default='cosine')
parser.add_argument('--pooling', type=str, help='Pooling layer to use.', default='average')
parser.add_argument('--top_k', type=int, help='Top k nodes to select.', default=15)
parser.add_argument('--n_hidden_decoder', type=int, help='Number of hidden units in the model.', default=15)
parser.add_argument('--n_linear_decoder', type=int, help='Number of linear layers in the model.', default=2)
parser.add_argument('--dropout', type=float, help='Dropout rate (1 - keep probability).', default=0.5)
parser.add_argument('--learning_rate', type=float, help='Learning rate of the optimization algorithm.', default=0.001)

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
    node_encodings_parser = NodeEncodingsParser()
    node_encodings_model = parser.get_encoding(args.endoding_type)
    embeddings_model = GCN(input_dim=args.n_nodes,
                           type='gcn',
                           n_hidden=args.n_hidden_encoder,
                           n_layers=args.n_layers,
                           dropout=args.dropout)
    forecast_model = SGNN(embedding=embeddings_model,
                          similarity=args.similarity,
                          pooling=args.pooling,
                          top_k=args.top_k, 
                          n_linear=args.n_linear_decoder, 
                          n_hidden=args.n_hidden_decoder,
                          dropout=args.dropout)
    
    # define optimizer
    opt = optim.Adam(forecast_model.parameters(), lr=args.learning_rate)

    # define loss functions
    

    # initialize tqdm
    pbar = tqdm(range(args.epochs), total=len(sim.n_simulations), desc=f"Running {args.model_name} model")
    epochs_loss = []
    epochs_predictions = []
    for epoch in pbar:

        batch_loss = []
        batch_predictions = []
        for batch in train_loader:

            opt.zero_grad()  

            # initialize loss variables = 0
            loss_tot = 0
            for i, j in enumerate(range(0, args.batch_size * 2, 2)):
            
                # get inputs
                x1 = batch.x[j, :, :]
                x2 = batch.x[j+1, :, :]

                # compute node encodings
                x1_enc = node_encodings_model.forward(x1)
                x2_enc = node_encodings_model.forward(x2)

                # computer node embeddings
                x1_enc_embeddings = embeddings_model1.forward(x1_enc)
                x2_enc_embeddings = embeddings_model2.forward(x2_enc)

                # forward pass
                pred_cov, z1, z2 = forecast_model.forward(x1_enc_embeddings, x2_enc_embeddings)

                # store pred and true values
                batch_predictions.append([pred_cov, batch[i].y])

                # compute loss functions
                loss = 0

                # add to initialized loss variables += loss functions
                loss_tot += loss

            ## backward pass
            loss_tot.backward()
            opt.step()

            ## store loss values
            batch_loss.append(loss_tot.detach().item())

        epochs_predictions.append(torch.tensor(batch_predictions))

        epochs_loss.append(torch.stack([torch.tensor(batch_loss), torch.tensor(batch_loss)], axis=1))

    # pred list to tensor
    epochs_predictions = torch.stack(epochs_predictions)
    epochs_loss = torch.stack(epochs_loss)

    pbar = tqdm(sim.n_simulations, total=len(sim.n_simulations), desc="Running SDNE0 model on test data")
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

                    # forward pass
                    pred_cov, z1, z2 = forecast_model.forward(x1, x2)

                    # save results
                    simulation_results.append([pred_cov, cov])
            
            simulation_results = torch.tensor(simulation_results)
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
        "train_loss": epochs_loss,
    }

    model_name = f'{args.model_name}_{int(args.n_hidden)}_{int(args.n_layers_enc)}_{int(args.n_layers_dec)}_{int(args.epochs)}'

    # check if file exists
    output_path = f"{os.path.dirname(__file__)}/data/outputs/{args.dataset_name}/{model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save file
    if args.sample:
        save_pickle(path=f"{output_path}/sample_args.pkl", obj=outargs)
        save_pickle(path=f"{output_path}/sample_predictions.pkl", obj=predictions)
        save_pickle(path=f"{output_path}/sample_training_info.pkl", obj=training_info)
        torch.save(model.state_dict(), f"{output_path}/model_sample.pth")
    else:
        save_pickle(path=f"{output_path}/args.pkl", obj=outargs)
        save_pickle(path=f"{output_path}/predictions.pkl", obj=predictions)
        save_pickle(path=f"{output_path}/training_info.pkl", obj=training_info)
        torch.save(model.state_dict(), f"{output_path}/model.pth")
