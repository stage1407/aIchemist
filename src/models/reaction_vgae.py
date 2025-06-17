import torch
from torch_geometric.nn import VGAE
from src.func.chem_structures import reaction_graph
import networkx as nx

class ReactionVGAE(VGAE):
    def __init__(self,encoder,input_dim,hidden_dim,edge_attr_dim,repr_learning=False,expected_feature_dim=18):
        super(ReactionVGAE, self).__init__(encoder)
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(2*input_dim + edge_attr_dim, hidden_dim), 
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_dim,edge_attr_dim)
        )

        self.rl = repr_learning
        if repr_learning:
            self.node_decoder = torch.nn.Sequential(
                torch.nn.Linear(encoder.hidden_dim, input_dim),
                torch.nn.ReLU()
            )
    
    def decode(self, z, edge_index, edge_attr):
        row, col = edge_index
        edge_features = torch.cat([z[row], z[col], edge_attr], dim=1)
        edge_out = self.edge_pred(edge_features)

        reaction_pred = nx.Graph()
        reaction_pred.add_nodes_from(range(len(z)))

        for i in range(edge_index.shape[1]):
            u,v = edge_index[0,i].item(), edge_index[1,i].item()
            reaction_pred.add_edge(u,v,bond_strength=edge_out[i].tolist())

        reaction_pred = reaction_graph(graph=reaction_pred)
        reaction_pred.x = z if not self.rl else self.node_decoder(z)
        reaction_pred.edge_attr = edge_attr
        reaction_pred.edge_index = torch.tensor(edge_index, dtype=torch.long)
        return reaction_pred