from torch_geometric.data import Dataset, Data
import torch

class ReactionDataset(Dataset):
    def __init__(self, mol_graphs, converter, transform=None, pre_transform=None, cache=False):
        super().__init__(None, transform, pre_transform)
        self.mol_graphs = mol_graphs    # List of mol_graph objects #TODO: Somehow the target graphs are missing
        self.converter = converter
        self.cache = cache

        if self.cache:
            print("Preprocessing and Caching of Data...")
            self.cached_data = []
            for idx, mg in enumerate(self.mol_graphs):
                input_data, target_data = self.converter.reaction_to_data(mg)
                self.cached_data.append((input_data, target_data))
                if idx % 10 == 0:
                    print(f"{idx} von {len(self.mol_graphs)} Graphen vorverarbeitet")
        else:
            self.cached_data = None
    
    def len(self):
        return len(self.mol_graphs)
    
    def __getitem__(self, idx):
        if self.cache:
            return self.cached_data[idx]
        else:
            data = self.mol_graphs[idx]
            # print("RD",data)
            input_data, target_data = self.converter.reaction_to_data(data)
            # print("Reaction,Educt", reaction_data, input_data)
            # TODO: Maybe they belong to this notation (target graphs instead of randomized targets) (learning the correlation between educt graph and product graph)
            # Add target attributes for node/edge labels (mock example)
            # data.node_target = torch.randint(0, 2, (data.x.size(0),))    # Binary node labels
            # data.edge_target = torch.randint(0, 3, (data.edge_index.size(1),))      # Edge change labels

            # return data
            return input_data, target_data