import torch
from torch_geometric.data import Data
import numpy as np

class MolGraphConverter:
    """
    Eine Klasse, um `mol_graph`-Objekte in PyTorch Geometric `Data`-Objekte zu konvertieren.
    Die Konvertierung unterstützt optionale Normalisierung und die Kodierung von Kantenfeatures als One-Hot-Vektoren.
    """

    def __init__(self, normalize_features=False, one_hot_edges=True):
        """
        Initialisiere den Konverter mit den gewünschten Optionen.
        :param normalize_features: Wenn True, werden die Knotenfeatures normalisiert.
        :param one_hot_edges: Wenn True, werden die Kantenfeatures als One-Hot-Vektoren kodiert.
        """
        self.normalize_features = normalize_features
        self.one_hot_edges = one_hot_edges

    def convert(self, mol_graph):
        """
        Konvertiere ein `mol_graph`-Objekt in ein PyTorch Geometric `Data`-Objekt.
        :param mol_graph: Das `mol_graph`-Objekt, das konvertiert werden soll.
        :return: Ein PyTorch Geometric `Data`-Objekt.
        """
        # Knotenfeatures extrahieren
        node_features = []
        for node_idx in mol_graph.nodes:
            feature_vector = mol_graph.nodes[node_idx]["feature"]
            node_features.append(feature_vector)

        node_features = torch.tensor(node_features, dtype=torch.float)

        # Optional: Normalisierung der Knotenfeatures
        if self.normalize_features:
            node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-9)

        # Kanteninformationen extrahieren
        edge_index = []
        edge_attr = []
        for edge in mol_graph.edges:
            edge_index.append([edge[0], edge[1]])

            # Extrahiere den numerischen Typ der Kante (Bindungstyp)
            bond_type = mol_graph.edges[edge]["bond_type"]
            if self.one_hot_edges:
                # Kodierung als One-Hot-Vektor
                max_bond_type = 3  # Angenommen: Einfach-, Doppel-, Dreifachbindung
                one_hot = [1 if i == bond_type else 0 for i in range(max_bond_type + 1)]
                edge_attr.append(one_hot)
            else:
                edge_attr.append([bond_type])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Erstelle ein PyTorch Geometric `Data`-Objekt
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        return data

# Beispielverwendung
# converter = MolGraphConverter(normalize_features=True, one_hot_edges=True)
# pyg_data = converter.convert(mol_graph)
