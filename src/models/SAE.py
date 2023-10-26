import torch
import torch.nn as nn
import torch.optim as optim

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        h = self.encoder(x)
        x_reconstructed = self.decoder(h)
        return x_reconstructed, h

class GraphModel(nn.Module):
    def __init__(self, S, D, layers_sizes, rho, beta, learning_rate, num_epochs):
        super(GraphModel, self).__init__()
        self.S = S
        self.D = D
        self.rho = rho
        self.beta = beta
        self.layers_sizes = layers_sizes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs    
        self.autoencoders = nn.ModuleList([SparseAutoencoder(layers_sizes[i], layers_sizes[i+1]) for i in range(len(layers_sizes) - 1)])
        
    def forward(self):
        X = torch.mm(torch.inverse(self.D), self.S)
        hidden_activations = []
        
        for autoencoder in self.autoencoders:
            optimizer = optim.Adam(autoencoder.parameters(), lr=self.learning_rate)
            for _ in range(self.num_epochs):
                x_reconstructed, h = autoencoder(X)
                loss = self.loss(x_reconstructed, X, h)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            X = h.detach()  # Set next input as the hidden activations
            hidden_activations.append(X)
        
        return hidden_activations
    
    def loss(self, x_reconstructed, x, h):
        mse_loss = nn.MSELoss()(x_reconstructed, x)
        kl_div_loss = self.kl_div(h)
        return mse_loss + self.beta * kl_div_loss
    
    def kl_div(self, h):
        rho_hat = torch.mean(h, dim=0)
        kl_div = self.rho * torch.log(self.rho / (rho_hat + 1e-10)) + (1 - self.rho) * torch.log((1 - self.rho) / (1 - rho_hat + 1e-10))
        return torch.sum(kl_div)
