import networkx as nx               #type: ignore
import matplotlib.pyplot as plt     #type: ignore
from rdkit import Chem

# Define atom colors based on their symbols
ATOM_COLORS = {
    'C': 'black',    # Carbon
    'O': 'red',      # Oxygen
    'N': 'blue',     # Nitrogen
    'H': 'gray',     # Hydrogen
    'S': 'yellow',   # Sulfur
    'P': 'orange',   # Phosphorus
    'F': 'green',    # Fluorine
    'Cl': 'green',   # Chlorine
    'Br': 'brown',   # Bromine
    'I': 'purple'    # Iodine
}

def visualize_molecule_graph(G: nx.Graph):
    """
    Visualizes a molecular graph with atom symbols as nodes and bond types as edge labels.

    :param G: A NetworkX graph representing the molecule.
    """
    if G.number_of_nodes() == 0:
        print("The graph is empty. Nothing to visualize.")
        return
    
    # Define a layout for the graph (spring layout is default, but can be changed)
    pos = nx.spring_layout(G)  # Arrange nodes with a spring layout
    # Extract atom types and use them as node labels
    #node_labels = nx.get_node_attributes(G, 'element')  # Node labels as atom symbols (e.g., C, N, O)
    #node_colors = [ATOM_COLORS.get(node_labels[node], 'black') for node in G.nodes()]  # Assign colors based on atom type
    node_labels = {node: f"{data['element']}{node}" for node, data in G.nodes(data=True)}
    node_colors = [ATOM_COLORS.get(data['element'], 'black') for _, data in G.nodes(data=True)]  # Assign colors based on atom type

    # Extract bond types as double values for edge labels
    edge_labels = nx.get_edge_attributes(G, 'bond_type')  # Get bond type as edge labels

    # Draw the graph with custom node colors and labels
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        labels=node_labels,            # Show atom symbols as node labels
        node_color=node_colors,        # Set node colors based on atom types
        with_labels=True,
        node_size=800,
        font_size=10,
        font_color='white',
        edge_color='black',
        width=2
    )

    # Draw bond type labels (BondTypeAsDouble)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='red')

    plt.title("Molecular Graph Visualization")
    plt.axis('off')
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Create an example molecule for Fischer indole synthesis intermediate
    smiles = 'C1=CC=CC=C1NN=C(C=O)C(C=O)'  # SMILES for the target molecule
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    G = nx.Graph()

    # Convert the molecule to a graph structure (add nodes and edges)
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), element=atom.GetSymbol())  # Add nodes with atom symbol as attribute

    # Add edges with bond type as a double value
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondTypeAsDouble())  # Add edges with bond type

    # Visualize the molecular graph
    visualize_molecule_graph(G)
