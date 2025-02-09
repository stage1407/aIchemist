import sys
from pathlib import Path
project_dir = Path(__file__).resolve().parent.parent.parent
sources = project_dir / "src"
sys.path.insert(0, str(sources))  # Nutze insert(0) statt append(), um Konflikte zu vermeiden
import numpy as np
from rdkit import Chem
import networkx as nx #type: ignore
from rdkit.Chem import rdFMCS
#from rdkit.Chem.rdMolDescriptors import CalcCrippenContribs
from itertools import product
from scipy.optimize import linear_sum_assignment
import importlib
mol_graph = importlib.import_module("src.func.mol_graph") 
mol_graph = getattr(mol_graph,"mol_graph")


class reaction_graph(nx.Graph):
    def __init__(self, mol_educts:mol_graph=None, mol_products:mol_graph=None, graph=None):
        assert (mol_educts is not None and mol_products is not None) or graph is not None 
        if mol_educts is not None and mol_products is not None:
            self.educts = mol_educts
            self.products = mol_products
            self.create_reaction_graph()
        elif graph is not None:
            self.add_edges_from(graph.edges(data=True))
        else:
            raise AttributeError("Could not create ReactionGraph by Nones as arguments")
        
    def create_reaction_graph(self):
        # Bipartite maximization of MCS relation between educts and products (extended backpacking problem) #! just a heuristic approach
        selected_pairs, _ = self.maximize_disjoint_mcs()
        
        # Derive atom-mapping
        atom_mapping = self.compute_atom_mapping(selected_pairs)

        # Compute bond changes
        bond_changes = self.compute_bond_changes(atom_mapping)

        # Create nodes and edges in reaction_graph based on bond_changes between educts and products
        self.build_graph_from_bond_changes(bond_changes)

    def maximize_disjoint_mcs(self):                 #! May be done by a GIN
        reactants = self.educts
        products = self.products
        def compute_mcs_sizes(r_list, p_list):
            """
            Berechnet paarweise die MCS-Größen zwischen Edukten und Produkten.
            Gibt eine Gewichtungsmatrix und die Paare zurück
            """
            pairs = []
            weights = []

            for (i, reactant),(j,product) in product(enumerate(r_list), enumerate(p_list)):
                # reactant = Chem.MolFromSmiles(reactant)
                # product = Chem.MolFromSmiles(product)

                # Compute Maximal Common Substructure
                mcs_result = rdFMCS.FindMCS([reactant,product])
                mcs_size = mcs_result.numAtoms  # Größe == Anzahl Atome
                
                # save the pair and weight
                pairs.append((reactant, product, mcs_result.smartsString, mcs_size))
                weights.append((i, j, mcs_size))
            return pairs, weights

        pairs, weights = compute_mcs_sizes(reactants.mol_list, products.mol_list)

        # number of educts and products
        num_reactants = len(reactants)
        num_products = len(products)

        # create weight matrix
        cost_matrix = np.zeros((num_reactants, num_products))
        for i,j,mcs_size in weights:
            cost_matrix[i,j] = -mcs_size                        # hungarian method, maximizing, by minimizing the negatives

        # solving the assignment problem
        reactant_indices, product_indices = linear_sum_assignment(cost_matrix)

        # Gathering maximal MCS and their pairs
        selected_pairs = []
        total_mcs_size = 0
        for r_idx, p_idx in zip(reactant_indices, product_indices):
            # get the belonging pair and MCS info
            for pair in pairs:
                if reactants[r_idx] == pair[0] and products[p_idx] == pair[1]:
                    selected_pairs.append(pair)
                    total_mcs_size += pair[3]
                    break

        return selected_pairs, total_mcs_size

    def compute_atom_mapping(self, selected_pairs):
        atom_mapping = {}
        for pair in selected_pairs:
            mcs_smarts = pair[2]

            reactant_mol : Chem.Mol = pair[0]
            product_mol : Chem.Mol = pair[1]

            mcs = Chem.MolFromSmarts(mcs_smarts)

            reactant_match = reactant_mol.GetSubstructMatch(mcs)
            product_match = product_mol.GetSubstructMatch(mcs)

            for r_idx, p_idx in zip(reactant_match, product_match):
                #? Is this enough for the condition
                if reactant_mol.GetAtomWithIdx(r_idx).GetSymbol() == product_mol.GetAtomWithIdx(p_idx).GetSymbol():
                    atom_mapping[r_idx] = p_idx

        return atom_mapping

    def compute_bond_changes(self, atom_mapping):
        bond_changes = {}
        # Compute this for each molecule
        atom_indices = []
        for mol in self.educts:
            mol : Chem.Mol = mol
            for atom in mol.GetAtoms():
                atom : Chem.Atom = atom
                atom_indices.append(atom.GetIdx())
        for idx1, idx2 in product(atom_indices, atom_indices):
            if idx1 != idx2:
                diff = self.products[atom_mapping[idx1]][atom_mapping[idx2]]["feature"][0] - self.educts[idx1][idx2]["feature"][0]
                if diff != 0:
                    e : set = {idx1,idx2}
                    bond_changes[e] = diff
        return bond_changes        

    def build_graph_from_bond_changes(self, bond_changes):
        """
        Baut den Reaktionsgraphen aus den Bindungsänderungen auf.
        """
        for (atom_idx1, atom_idx2), bond_change in bond_changes.items():
            self.add_edge(atom_idx1, atom_idx2, weight=bond_change)

    def chemical_distance(self):
        """
        Computes the chemical Distance between self and other_graph.
        """
        # Sums up
        sum_g1 = sum(data["weight"] for _,_,data in self.educts.edges(data=True))
        sum_g2 = sum(data["weight"] for _,_,data in self.products.edges(data=True))
        
        selected_pairs = self.maximize_disjoint_mcs()
        sum_mcs = 0
        for pair in selected_pairs:
            mcs_mol = Chem.MolFromSmarts(pair[2])
            for bond in mcs_mol.GetBonds():
                sum_mcs += int(bond.GetBondTypeAsDouble())
                

        chemical_distance = sum_g1 + sum_g2 - 2*sum_mcs

        return chemical_distance