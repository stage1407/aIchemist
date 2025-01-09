import torch
import torch.nn as nn
import torch.nn.functional as F

class GINLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x, edge_index):
        # x: [num_nodes, in_features]
        # edge_index: [2, num_edges]

        # Aggregate messages from neighbors
        row, col = edge_index
        aggregated_messages = torch.scatter_add(self.mlp(x[col]), 0, index=row)

        # Update node representations
        out = self.mlp(x + aggregated_messages)
        return out

class GIN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GINLayer(in_features, hidden_features))
            in_features = hidden_features

        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.final_layer(x)
        return x