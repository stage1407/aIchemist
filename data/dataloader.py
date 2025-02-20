from torch_geometric.data import Dataset
import torch

class ReactionDataset(Dataset):
    def __init__(self, mol_graphs, converter, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.mol_graphs = mol_graphs    # List of mol_graph objects #TODO: Somehow the target graphs are missing
        self.converter = converter
    
    def len(self):
        return len(self.mol_graphs)
    
    def __getitem__(self, idx):
        mol_graph = self.mol_graphs[idx]
        data, educt_graph = self.converter.reaction_to_data(mol_graph)
        # TODO: Maybe they belong to this notation (target graphs instead of randomized targets) (learning the correlation between educt graph and product graph)
        # Add target attributes for node/edge labels (mock example)
        data.node_target = torch.randint(0, 2, (data.x.size(0),))    # Binary node labels
        data.edge_target = torch.randint(0, 3, (data.edge_index.size(1),))      # Edge change labels

        return data, educt_graph