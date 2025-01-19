import torch
from torch_geometric.nn import GAE

class ReactionGAE(GAE):
    def __init__(self, encoder):
        super(ReactionGAE, self).__init__(encoder)
        self.edge_pred = torch.nn.Linear(encoder.hidden_dim * 2 + encoder.edge_attr_dim, 1)

    def decode(self, z, edge_index, edge_attr):
        row, col = edge_index
        edge_features = torch.cat([z[row], z[col], edge_attr], dim=1)
        edge_out = self.edge_pred(edge_features)
        return edge_out