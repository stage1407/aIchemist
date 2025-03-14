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

def mol_graph_to_data(mg : mol_graph):
    node_indices = sorted(mg.nodes)
    features = [mg.nodes[i]["feature"] for i in node_indices]
    x = torch.tensor(features, dtype=torch.float)
    edges = list(mg.edges)
    if len(edges) > 0:
        edge_index = torch.tensor([[u,v] for u,v in edges], dtype=torch.long).t().contiguous()
        print(edges)
        edge_attr_list = [mg.edges[e]["feature"] for e in edges]
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,0),dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

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

    """def convert_to_data(self, r_graph : reaction_graph, educt_graph, product_graph):
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
            # print("Feature Educt:",educt_graph.nodes[node_idx]["feature"])
            educt_features = torch.tensor(educt_graph.nodes[node_idx]["feature"], dtype=torch.float) \
                if node_idx in educt_graph.nodes else torch.zeros(len(product_graph.nodes[next(iter(product_graph.nodes))]["feature"]))
            product_features = torch.tensor(product_graph.nodes[atom_mapping[node_idx]]["feature"], dtype=torch.float) \
                if atom_mapping[node_idx] in product_graph.nodes else torch.zeros(len(educt_features))

            # Compute feature change (Product - Educt)
            feature_vector = product_features - educt_features
            node_features.append(feature_vector)

        # print("Tensor:", node_features)
        if not node_features:
            node_features = torch.zeros(1, len(product_graph.nodes[next(iter(product_graph.nodes))]["feature"]))  # Safe fallback tensor
        else:
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
        return data"""

    def reaction_graph_to_data(r_graph : reaction_graph, educt_graph : mol_graph, product_graph : mol_graph):
        node_indices = sorted(r_graph.nodes)
        node_index_map = {}
        diff_features = []
        for i, node in enumerate(node_indices):
            node_index_map[node] = i
            if node in educt_graph.nodes and r_graph.bijection.get(node) in product_graph.nodes:
                educt_feat = torch.tensor(educt_graph.nodes[node]["feature"], dtype=torch.float)
                prod_feat = torch.tensor(product_graph.nodes[r_graph.bijection[node]]["feature"], dtype=torch.float)
                diff_feat = prod_feat - educt_feat
            else:
                # Falls keine Zuordnung existiert, Standardvektor (z.B. Nullen) verwenden
                feat_dim = len(next(iter(educt_graph.nodes.values()))["feature"])
                diff_feat = torch.zeros(feat_dim, dtype=torch.float)
            diff_features.append(diff_feat)
        x = torch.stack(diff_features)

        # 2. Kanteninformationen
        edge_index_list = []
        edge_attr_list = []
        for edge in r_graph.edges:
            u, v = edge
            # Verwende den sequenziellen Index, den wir für r_graph-Knoten vergeben haben
            if u in node_index_map and v in node_index_map:
                edge_index_list.append([node_index_map[u], node_index_map[v]])
                # Nun: Falls in educt_graph und product_graph die entsprechenden Kanten existieren,
                # berechne den Unterschied der Kantenfeatures.
                # Hier musst du anpassen, wie deine Kantenfeatures gespeichert sind.
                educt_edge = educt_graph.edges.get((u, v), None)
                prod_edge = product_graph.edges.get((r_graph.bijection.get(u), r_graph.bijection.get(v)), None)
                if educt_edge is not None and prod_edge is not None:
                    # Beispiel: Es wird angenommen, dass in beiden Graphen unter "feature" ein Vektor steht.
                    educt_edge_feat = educt_edge.get("feature", None)
                    prod_edge_feat = prod_edge.get("feature", None)
                    m = len(educt_edge_feat)
                    n = len(prod_edge_feat)
                    assert m == n
                    if educt_edge_feat is not None and prod_edge_feat is not None:
                        # Berechne die Differenz (Elementweise)
                        diff_edge_feat = [p - e for p, e in zip(prod_edge_feat, educt_edge_feat)]
                    else:
                        # Fallback: Vektor aus Nullen, Länge wie gewünscht (hier beispielhaft 1)
                        diff_edge_feat = [0]*n
                else:
                    # Falls keine Kanten in beiden Graphen existieren, definiere einen Standard
                    diff_edge_feat = [0]*10             #! Hard-Coded: Needs refactoring
                edge_attr_list.append(diff_edge_feat)
        print(edge_attr_list)
        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 0), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    
    def reaction_to_data(self, react_data):
        rd = react_data
        scalar = 0
        # print("Scaling Reaction Data...")
        min_num = min(rd["educt_amounts"] + rd["product_amounts"] + [100])      # TODO: Catch
        scalar = 1/min(rd["educt_amounts"] + rd["product_amounts"] + [100])     # TODO: This
        list_ed = zip(list(rd["educts"]),list(rd["educt_amounts"]))
        smilies_ed = []
        for smiles,num in list_ed:
            amount = 1 if num == min_num else int(num*scalar)               #NaIve Scale
            # print(amount)
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
        print("ReactionGraph:",react)
        #print(self, react, ed, prod)
        # return self.convert_to_data(react, ed, prod)
        ed_data = mol_graph_to_data(ed)
        react_data = MolGraphConverter.reaction_graph_to_data(react, ed, prod)
        return ed_data,react_data