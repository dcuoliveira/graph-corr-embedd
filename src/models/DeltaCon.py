import torch
from netrd.distance import DeltaCon

class DeltaConWrapper(DeltaCon):
    def __init__(self):
        super(DeltaConWrapper, self).__init__()

    def forward(self, G1, G2):
        """Compute the DeltaCon similarity score based on the similarity matrix S."""
        dis = self.dist(G1=G1, G2=G2)
        
        return dis


if __name__ == "__main__":
    pass