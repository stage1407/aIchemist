import torch
from torch_geometric.data import Data

def generate_mock_mol_graph(num_nodes=10, num_edges=20, feature_dim=5, edge_attr_dim=3):
    """
    Generiert einen Mock-Molekülgraphen mit zufälligen Knoten- und Kantenfeatures.
    :param num_nodes: Anzahl der Knoten im Graphen
    :param num_edges: Anzahl der Kanten im Graphen
    :param feature_dim: Dimension der Knotenfeatures
    :param edge_attr_dim: Dimension der Kantenfeatures
    :return: Ein PyTorch Geometric Data-Objekt
    """
    # Knotenfeatures
    x = torch.rand((num_nodes, feature_dim))

    # Zufällige Kanten generieren
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Kantenfeatures
    edge_attr = torch.rand((num_edges, edge_attr_dim))

    # Zufällige Labels für Knoten und Kanten
    node_target = torch.randint(0, 2, (num_nodes,))  # Binäre Labels für Knoten
    edge_target = torch.randint(-2, 3, (num_edges,))  # Labels für Kanten (z. B. Bindungsänderungen)

    # Erstelle das Data-Objekt
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_target = node_target
    data.edge_target = edge_target

    return data

def generate_mock_dataset(num_graphs=100, num_nodes=10, num_edges=20, feature_dim=5, edge_attr_dim=3):
    """
    Generiert ein Mock-Dataset aus mehreren zufälligen Molekülgraphen.
    :param num_graphs: Anzahl der Graphen im Dataset
    :param num_nodes: Anzahl der Knoten pro Graph
    :param num_edges: Anzahl der Kanten pro Graph
    :param feature_dim: Dimension der Knotenfeatures
    :param edge_attr_dim: Dimension der Kantenfeatures
    :return: Liste von PyTorch Geometric Data-Objekten
    """
    dataset = []
    for _ in range(num_graphs):
        graph = generate_mock_mol_graph(num_nodes, num_edges, feature_dim, edge_attr_dim)
        dataset.append(graph)
    return dataset

if __name__ == "__main__":
    # Beispiel für die Generierung von Mock-Daten
    mock_dataset = generate_mock_dataset(num_graphs=10, num_nodes=15, num_edges=30, feature_dim=8, edge_attr_dim=4)
    print(f"Generierte {len(mock_dataset)} Mock-Graphen.")
    print(f"Erster Graph: {mock_dataset[0]}")
