
from node_encodings.DegreeEncoding import DegreeEncoding
from node_encodings.IdentityEncoding import IdentityEncoding
from node_encodings.LaplacianEncoding import LaplacianEncoding
from node_encodings.RandomWalkEncoding import RandomWalkEncoding

class NodeEncodingsParser:
    def __init__(self):
        self.encodings = {
            "degree": DegreeEncoding,
            "identity": IdentityEncoding,
            "laplacian": LaplacianEncoding,
            "random_walk": RandomWalkEncoding
        }

    def get_encoding(self, encoding_type, **params):
        """
        Initialize and return the appropriate encoding class based on encoding_type.

        Parameters:
        encoding_type (str): Type of the encoding (e.g., 'degree', 'identity', 'laplacian', 'random_walk')
        **params: Additional parameters required for initializing the encoding class

        Returns:
        object: An instance of the appropriate encoding class
        """
        if encoding_type not in self.encodings:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        encoding_class = self.encodings[encoding_type]
        return encoding_class(**params)