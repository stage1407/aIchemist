import torch
import torch.nn.functional as F
from src.func.chem_structures import reaction_graph
from torch_geometric.nn import GCNConv
import networkx as nx

class GNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_attr_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, input_dim)

        self.edge_predictor = torch.nn.Sequential(torch.nn.Linear(2*input_dim+edge_attr_dim, hidden_dim), 
                                                  torch.nn.ReLU(), 
                                                  torch.nn.Linear(hidden_dim,edge_attr_dim))
        self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, input_dim)
            )

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.output_layer(x)

        edge_input = torch.cat([x[edge_index[0]],x[edge_index[1]], edge_attr], dim=1)
        edge_attr = self.edge_predictor(edge_input)

        reaction_net = nx.Graph()
        reaction_net.add_nodes_from(range(len(x)))

        for i in range(edge_index.shape[1]):
            u, v = edge_index[0,i].item(), edge_index[1,i].item()
            reaction_net.add_edge(u,v,bond_strength = edge_attr[i].tolist())

        reaction_pred = reaction_graph(graph=reaction_net)
        reaction_pred.x = x
        reaction_pred.edge_attr = edge_attr
        return reaction_pred