import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_attr_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.edge_attr_dim = edge_attr_dim
        self.edge_predictor = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        y = (x[edge_index[0]] + x[edge_index[1]]) / 2
        y = self.edge_predictor(y)
        return x,y