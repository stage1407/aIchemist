import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class ReactionGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_attr_dim, heads=4, dropout=0.0):
        super(ReactionGAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)

        # Knoten-Vorhersage
        self.node_pred = torch.nn.Linear(hidden_dim, output_dim)
        # Kanten-Vorhersage
        self.edge_pred = torch.nn.Linear(hidden_dim * 2 + edge_attr_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # GAT Message Passing
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))

        # Knoten-Vorhersagen
        node_out = self.node_pred(x)

        # Kantenvorhersagen
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_out = self.edge_pred(edge_features)

        return node_out, edge_out