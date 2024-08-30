import torch
import numpy as np

class LossEigen:
    def __init__(self, loss_type: str="norm"):
        self.loss_type = loss_type

    def forward(self, adj, x):
        # Convert tensors to numpy arrays if they are PyTorch tensors
        if isinstance(adj, torch.Tensor):
            adj = adj.detach().cpu().numpy()
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        # Compute the eigenvalues of the adjacency matrices
        eigenvalues_adj = np.linalg.eigvals(adj)
        eigenvalues_x = np.linalg.eigvals(x)

        # Sort the eigenvalues in ascending order
        eigenvalues_adj = np.sort(eigenvalues_adj)
        eigenvalues_x = np.sort(eigenvalues_x)
        
        # Compute the pairwise Euclidean distance between the eigenvalues
        if self.loss_type == "norm":
            distance = np.linalg.norm(eigenvalues_adj - eigenvalues_x)
        elif self.loss_type == "dot":
            distance = np.dot(eigenvalues_adj, eigenvalues_x)
        elif self.loss_type == "mse":
            distance = np.mean((eigenvalues_adj - eigenvalues_x)**2)
        
        # Convert the distance to a PyTorch tensor
        loss = torch.tensor(distance, dtype=torch.float32)
        
        return loss

DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        
        adj_tensor = torch.tensor(adj, dtype=torch.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        
        loss = LossEigen(adj_tensor, x_tensor)
        print("Loss:", loss.item())