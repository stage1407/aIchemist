import torch
import torch.nn as nn
import torch.nn.functional as F

class AtomMappingGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AtomMappingGNN, self).__init__()
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.gnn = nn.Linear(hidden_dim, hidden_dim)
        self.mapping_predictor = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, educt_graph, product_graph):
        # Encode node features #TODO: find a better function?
        #! Needs a sanatizer C must maps to C (RGAT?)
        educt_nodes = F.relu(self.node_encoder(educt_graph.x))
        product_nodes = F.relu(self.node_encoder(product_graph.x))

        # GNN Message Passing
        educt_nodes = self.gnn(educt_nodes)
        product_nodes = self.gnn(product_nodes)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(educt_nodes, product_nodes.T)

        # Enforce elementary constraint

        atom_mapping_probs = F.softmax(similarity_matrix, dim=1)

        return atom_mapping_probs