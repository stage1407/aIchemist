import sys
from pathlib import Path
project_dir = Path(__file__).resolve().parent.parent.parent
sources = project_dir / "src"
sys.path.insert(0, str(sources))
from src.func.chem_structures import mol_graph, reaction_graph, properties
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
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

    def convert_to_data(self, r_graph : reaction_graph, educt_graph, product_graph):
        """
        Converts a `r_graph` into a PyTorch Geometric `Data` object, using atomic properties
        from the product graph MINUS the atomic properties from the educt graph.

        :param reaction_graph: The `r_graph` instance representing bond changes in a reaction.
        :param educt_graph: The `mol_graph` of the reactants.
        :param product_graph: The `mol_graph` of the products.
        :return: A PyTorch Geometric `Data` object.
        """
        # Extract Node Features (Atomic Property Changes)
        node_features = []
        node_index_map = {}  # Maps reaction graph node index to sequential index for PyTorch Geometric

        # Simulating empty graph
        if r_graph.isEmpty():
            # print("True")
            r_graph.add_node(0,feature=len(properties)*[0])

        for i, node_idx in enumerate(r_graph.nodes):
            node_index_map[node_idx] = i
            atom_mapping = r_graph.bijection

            # Get atomic properties from educt and product graphs
            print("Feature Educt:",educt_graph.nodes[node_idx]["feature"])
            educt_features = torch.tensor(educt_graph.nodes[node_idx]["feature"], dtype=torch.float) \
                if node_idx in educt_graph.nodes else torch.zeros(len(product_graph.nodes[next(iter(product_graph.nodes))]["feature"]))
            product_features = torch.tensor(product_graph.nodes[atom_mapping[node_idx]]["feature"], dtype=torch.float) \
                if atom_mapping[node_idx] in product_graph.nodes else torch.zeros(len(educt_features))

            # Compute feature change (Product - Educt)
            feature_vector = product_features - educt_features
            node_features.append(feature_vector)

        print("Tensor:", node_features)
        try:
            node_features = torch.stack(node_features)
        except Exception as e:
            def print_full_error(e):
                tb = e.__traceback__
                print(f"Exception Type: {type(e).__name__}")
                print(f"Exception Message: {str(e)}")

                while tb:  # Walk through traceback manually
                    print(f"    File: {tb.tb_frame.f_code.co_filename}, Line: {tb.tb_lineno}, in {tb.tb_frame.f_code.co_name}")
                    tb = tb.tb_next
            print_full_error(e)
            print(f"Args: {e.args}")

        # Normalize Features if enabled
        if self.normalize_features:
            node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-9)

        # Extract Edge Information (Bond Changes)
        edge_index = []
        edge_attr = []
        for edge in r_graph.edges:
            node1, node2 = edge
            bond_change = r_graph.edges[edge]["weight"]  # Bond change stored in r_graph

            # Convert to sequential index
            if node1 in node_index_map and node2 in node_index_map:
                edge_index.append([node_index_map[node1], node_index_map[node2]])

                # One-Hot Encoding of Bond Change if enabled
                if self.one_hot_edges:
                    max_bond_change = 3  # Assume bond changes range from -3 (triple bond broken) to +3
                    one_hot = [1 if i == bond_change else 0 for i in range(-max_bond_change, max_bond_change)]  # Range (-3 to 3)
                    edge_attr.append(one_hot)
                else:
                    edge_attr.append([bond_change])

    # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        return data, from_networkx(educt_graph)
    
    def reaction_to_data(self, react_data):
        rd = react_data
        scalar = 0
        print("Scaling Reaction Data...")
        min_num = min(rd["educt_amounts"] + rd["product_amounts"] + [100])      # TODO: Catch
        scalar = 1/min(rd["educt_amounts"] + rd["product_amounts"] + [100])     # TODO: This
        list_ed = zip(list(rd["educts"]),list(rd["educt_amounts"]))
        smilies_ed = []
        for smiles,num in list_ed:
            amount = 1 if num == min_num else int(num*scalar)               #NaIve Scale
            print(amount)
            smilies_ed += amount*[smiles]
        ed = mol_graph(smilies=smilies_ed)
        smilies_pr = []
        list_pr = zip(rd["products"],rd["product_amounts"])
        for smiles,num in list_pr:
            amount = 1 if num == min_num else int(num*scalar)               #NaIve Scale
            smilies_pr += amount*[smiles]
        prod = mol_graph(smilies=smilies_pr)
        #print("Scaling",smilies_ed,smilies_pr)
        react = reaction_graph(ed,prod)
        #print("Test:",react)
        #print(self, react, ed, prod)
        return self.convert_to_data(react, ed, prod)
