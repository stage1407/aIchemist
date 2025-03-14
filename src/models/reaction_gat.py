import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from src.func.chem_structures import reaction_graph
import networkx as nx

class ReactionGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_attr_dim, heads=4, dropout=0.0, expected_feature_dim=18):
        super(ReactionGAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)

        # Knoten-Vorhersage
        # Kanten-Vorhersage
        self.edge_pred = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2 + edge_attr_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, edge_attr_dim)
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # GAT Message Passing
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))

        # Knoten-Vorhersagen
        x = self.output_layer(x)

        # Kantenvorhersagen

        row, col = edge_index

        edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_out = self.edge_pred(edge_input)

        # Konstruktion des Reaktionsgraphen

        reaction_pred = nx.Graph()
        reaction_pred.add_nodes_from(range(len(x)))

        for i in range(edge_index.shape[1]):
            u, v = edge_index[0,i].item(), edge_index[1,i].item()
            reaction_pred.add_edge(u,v,bond_strength=edge_out[i].tolist())

        reaction_pred = reaction_graph(graph=reaction_pred)
        reaction_pred.x = x
        reaction_pred.edge_attr = edge_out
        return reaction_pred