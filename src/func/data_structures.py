import numpy as np
import json
from rdkit import Chem
import typing as type
import networkx as nx #type: ignore
from rdkit.Chem import AllChem 
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from rdkit.Chem import rdMolTransforms
#from rdkit.Chem.rdMolDescriptors import CalcCrippenContribs
import pubchempy as pcp #type: ignore
from enum import Enum
import os

MAX_DEPTH = 10
quantum_mechanics = False       # set True for modeling pi-electrons        #? Can be learned by Machine Learning?
geometric_properties = False    # set True for modeling hydrogen bonds      #? Can be learned by Machine Learning?
properties = [
    "Hybridization",
    "Formal charge",
    "Implicit hydrogens",
    "Aromaticity",
    "Chirality/Stereochemistry",
    "Explicit valence",
    "Implicit valence",
    "Lone pairs",
    "Bonding",
    "No lone pairs",
    "Degree",
    "Total_Degree",
    "Formal_Charge",
    "Lone_Pair_Count",
    "Aromatic",
    "Electronegativity",
    "Stereochemistry",
    "Hydrophobicity"
]

#*node_features = ["element","degree","charge","valence","lones","aromaticity","hydrogen","hybridization"]
#*edge_features = []

with open("./src/func/chemical_features.json","r") as file:
    feature_filter : dict = json.load(file)

properties = {
    "node_features": list(feature_filter["node_features"].keys()),
    "edge_features": list(feature_filter["edge_features"].keys()),
    "graph_features": list(feature_filter["graph_features"].keys())
}
def get_atom_properties(atom, property):
    node_properties = properties["node_features"]
    property_dict = {
        node_properties[0]: atom.GetHybridization().name,  # Returns the hybridization type (e.g., SP2).
        node_properties[1]: atom.GetFormalCharge(),  # Returns the formal charge on the atom.
        node_properties[2]: atom.GetNumImplicitHs(),  # Returns the number of implicit hydrogens.
        node_properties[3]: atom.GetIsAromatic(),  # Returns True if the atom is aromatic.
        node_properties[4]: atom.GetChiralTag().name,  # Returns chirality (e.g., R/S).
        node_properties[5]: atom.GetExplicitValence(),  # Returns the explicit valence of the atom.
        node_properties[6]: atom.GetImplicitValence(),  # Returns the implicit valence of the atom.
        node_properties[7]: (atom.GetTotalNumHs() + atom.GetImplicitValence() + atom.GetExplicitValence() - atom.GetTotalValence()) // 2, # Calculated manually as lone pairs are not directly accessible.
        node_properties[8]: [bond.GetBondType().name for bond in atom.GetBonds()],  # Returns bond types connected to the atom.
        node_properties[9]: ((atom.GetTotalNumHs() + atom.GetImplicitValence() + atom.GetExplicitValence() - atom.GetTotalValence()) // 2) == 0,  # True if no lone pairs.
        node_properties[10]: atom.GetDegree(),  # Returns the number of directly bonded atoms
        node_properties[11]: atom.GetTotalDegree(),  # Returns total degree including implicit hydrogens.
        node_properties[12]: atom.GetFormalCharge(),  # Same as Formal charge.
        node_properties[13]: (atom.GetTotalNumHs() + atom.GetImplicitValence() + atom.GetExplicitValence() - atom.GetTotalValence()) // 2,  # Same as Lone pairs.
        node_properties[14]: atom.GetIsAromatic(),  # Same as Aromaticity.
        node_properties[15]: atom.GetAtomicNum(),  # Approximate electronegativity by atomic number.
        node_properties[16]: atom.GetChiralTag().name,  # Same as Chirality/Stereochemistry.
        node_properties[17]: atom.GetAtomicNum() in [6, 7, 8, 15, 16, 17],  # An approximate way to infer hydrophobicity.
    }
    return property_dict[property]

def get_bond_properties(mol, bond, p):
    properties = {
        "Bond_Order": bond.GetBondTypeAsDouble(),  # Returns the bond order (e.g., 1.0, 2.0).
        "Electrophilic_Character": bond.GetBeginAtom().GetAtomicNum() in [6, 7] and bond.GetEndAtom().GetAtomicNum() in [8, 16],  # Approximate electrophilic character.
        "Nucleophilic_Character": bond.GetBeginAtom().GetAtomicNum() in [15, 16] or bond.GetEndAtom().GetAtomicNum() in [15, 16],  # Approximate nucleophilic character.
        #"Length": rdMolTransforms.GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),  # Bond length from 3D coordinates.
        "Aromaticity": bond.GetIsAromatic(),  # Checks if the bond is aromatic.
        #"Orientation": "Not directly available",  # Orientation calculation requires vector math.
        "Stereo_State": bond.GetStereo().name,  # Returns the stereochemistry of the bond.
        #"Distance": rdMolTransforms.GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())  # Same as Length.
    }
    return properties[p]

class mol_graph(nx.Graph):
    def __init__(self, smilies : str = None, mols : Chem.RWMol = None):
        """
        Initialize the MoleculeGraph from a SMILES string or RDKit molecule object.
        
        Parameters:
        - smiles: Optional; SMILES string representing the molecule.
        - mol: Optional; RDKit Mol object. If both smiles and mol are provided, mol will be used.
        """
        super().__init__()
        if (mols is None and smilies is None) or (mols is not None and smilies is not None):
            raise ValueError("Either a valid RDKit molecule or SMILES string must be provided.")
        self.smiles = []
        self.mols = []
        # read all smiles strings of the reaction
        if mols is None:
            for smiles in smilies:
                self.smiles.append(smiles)
                mol : Chem.rdchem.Mol = Chem.MolFromSmiles(smiles)
                self.mols.append(mol.AddHs())
        else:
            self.mols = mols
        # Make all hydrogen atoms in the molecule explicit
        # look up the config
        if geometric_properties:
            AllChem.EmbedMolecule(self.mol)
            AllChem.UFFOptimizeMolecule(self.mol)
        self._build_graph()

    def _build_graph(self):
        """
        Internal method to build the graph based on the RDKit molecule.
        - Nodes represent atoms.
        - Edges represent bonds with special handling for aromatic bonds.
        """
        self.phi : dict = {}
        index_shift = 0
        for mol in self.mols:
            i = 0
            for atom in mol.GetAtoms():
                self.add_node(atom.GetIdx() + index_shift, element=atom.GetSymbol(), feature=_derive_feature_vector(node=atom))
                i += 1
            for bond in mol.GetBonds():
                self.add_edge(bond.GetBeginAtomIdx() + index_shift, bond.GetEndAtomIdx() + index_shift, bond_type=bond.GetBondTypeAsDouble(), feature=_derive_feature_vector(edge=bond,mol=mol))
                self.phi.update({(bond.GetBeginAtomIdx() + index_shift, bond.GetEndAtomIdx() + index_shift): bond.GetBondTypeAsDouble()})
            index_shift = i
            if geometric_properties:
                self._add_hydrogen_bonds(mol)      #? feature or algorithmic retrieval

    def _add_hydrogen_bonds(self, mol : Chem.RWMol, distance_threshold=2.5):
        #TODO: Comment
        self.donor_acceptor_pairs = self._find_donor_acceptor_pairs(mol)
        conf = mol.GetConformer()
        #TODO: Comment
        for donor, acceptor in self.donor_acceptor_pairs:
            donor_hydrogen = donor['hydrogen']
            acceptor_idx = acceptor.GetIdx()
            #TODO: Comment
            donor_hydrogen_coords = conf.GetAtomPosition(donor_hydrogen.GetIdx())
            acceptor_coords = conf.GetAtomPosition(acceptor_idx)
            #TODO: Comment
            distance = donor_hydrogen_coords.Distance(acceptor_coords)
            #TODO: Comment
            if distance <= distance_threshold:
                self.add_edge(donor_hydrogen.GetIdx(), acceptor_idx, [0.1,False,Chem.rdchem.BondStereo.STEREONONE, distance])
    
    def _find_donor_acceptor_pairs(self, mol : Chem.Mol):
        donors = []
        acceptors = []
        #TODO: Comment
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in (7,8,9): # N, O, F
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1: # Hydrogen
                        donors.append({'atom': atom, 'hydrogen': neighbor})
            if atom.GetAtomicNum() in (7, 8, 9) and atom.GetTotalDegree() <= 3:
                acceptors.append(atom)
        #TODO: Comment
        donor_acceptor_pairs = []
        for donor in donors:
            for acceptor in acceptors:
                if donor['atom'].GetIdx() != acceptor.GetIdx():
                    donor_acceptor_pairs.append((donor, acceptor))
        return donor_acceptor_pairs

    def get_atom_element(self, atom_idx):
        if atom_idx not in self.nodes:
            raise ValueError(f"Atom index {atom_idx} not found in the graph.")
        return self.nodes[atom_idx]["element"]

    def get_elements(self):
        elem_list = []
        for node_id in self.nodes:
            elem_list.append(self.get_atom_element(node_id))
        return elem_list

    def get_atom_features(self, atom_idx):
        """
        Retrieve atom features from the graph.
        
        Parameters:
        - atom_idx: Index of the atom node.
        
        Returns:
        - A dictionary of features for the specified atom.
        """
        if atom_idx not in self.nodes:
            raise ValueError(f"Atom index {atom_idx} not found in the graph.")
        return self.nodes[atom_idx]["feature"]
    
    def get_bond_features(self, atom_idx1, atom_idx2):
        """
        Retrieve bond features between two atoms.

        Parameters:
        - atom_idx1: Index of the first atom.
        - atom_idx2: Index of the second atom.

        Return:
        - A dictionary of features for the bond.
        """
        if not self.has_edge(atom_idx1, atom_idx2):
            raise ValueError(f"No bond exists between atom {atom_idx1} and atom {atom_idx2}.")
        return self.edges[atom_idx1, atom_idx2]["feature"]
    
    def get_substructural_features(self, subgraph : nx.Graph):
        #TODO:
        pass

def _derive_feature_vector(node=None, edge=None, subgraph=None, mol=None):
    features = []
    if node:
        feature_definition : dict = feature_filter["node_features"]
        for p in properties["node_features"]:
            if node.GetSymbol() in feature_definition[p]:
                features.append(get_atom_properties(node,p))
    elif edge and mol:
        feature_definition : dict = feature_filter["edge_features"]
        bond : Chem.Bond = edge
        for p in properties["edge_features"]:
            if feature_definition[p]:
                features.append(get_bond_properties(mol,bond,p))
    elif subgraph and not subgraph:    #! Just to skip subgraph features yet!
        #? How to implement subgraphs?
        ("mol_interpretation")
        ("mol_fragment_type")
    else:
        raise AttributeError
    return features

class reaction_graph:
    #TODO: Implement this data structure
    def __init__(self, psi : dict): #, educts : mol_graph = None, products=None): 
        if psi is not None:
            assert sum(psi.values()) == 0
            self.psi = psi
        else:
            pass
            #print(psi)
            #self._init2(educts=educts, products=products)
    """
    def _init2(self, educts : mol_graph, products : mol_graph):
    """
    """
        Represents the graph theoretical difference: Product - Educt 

        Parameters:
        - educts: represents all educts, in a mol_graph
        - products: represents all products, in a mol_graph
        
        Returns:
        - reaction_graph
    """
    """
        # assert np.sort(educts.get_elements()) == np.sort(products.get_elements())
        psi = {}
        psi.setdefault(0)
        # It's a kind of magic (Eeeeeeeeh-Oohhhhhhh)
        max_mcs = None
        for i,e_mol in enumerate(educts.mols):
            max = 1
            for j,p_mol in enumerate(products.mols):
                mcs : rdFMCS.MCSResult = rdFMCS.FindMCS([e_mol, p_mol])
                mcs_mol = Chem.MolFromSmarts(mcs)    # Computing mcs over reaction, to know what is not changing in reaction
                e_mol_matching = e_mol.GetSubstructMatch(mcs_mol)
                p_mol_matching = p_mol.GetSubstructMatch(mcs_mol)
                if mcs.numAtoms > max:
                    max = mcs.numAtoms
                    max_mcs = (i,j,mcs_mol,e_mol_matching,p_mol_matching)
        #TODO: How to use pairwise mcs for reaction graphs


        assert sum(psi.values()) == 0       # feasibility
        self.psi = psi
    """
        
    def _algorithmic_constructed_reaction_graph(self, educts : mol_graph, products : mol_graph):
        """
        Cite pseudo-code of the reaction-network paper (drive)
        """
        i = 0
        i_max = MAX_DEPTH
        path_stack = []
        U = [[educts]]
        cond = False
        while not cond:
            if len(U[i]) != 0:
                G = U[i].pop()          #TODO: arbitrary graph
                path_stack.append(G)
                cd = compute_chemical_distance(educts, products)
                if cd == 0:
                    #* Output the molecular_manipulation path
                    pass
                else:
                    if i < i_max:
                        i += 1
                        F = self._compute_all_isomeric_graphs(G)
                        U.append()
                        #!RGAT part
                        #TODO:  implement feasibility 
                        #* U[i] = set of all graphs produced by feasible transformations
                        #* and a descending chemical distance
                        #? HOW TO?!
            else:
                i -= 1
                path_stack.pop()
            cond = i == 0
        
    def _compute_all_isomeric_graphs(self, G, descending_chemical_distance=True, feasible_transformations=True):
        
        pass
                
    def _construct_transformation(self, G):
        
        pass


    def __add__(self, other):
        if isinstance(other, reaction_graph):
            #TODO: Provide composition functionality via (+)
            pass
        return TypeError(f"Input has to be instance of the class: reaction_graph.")

def compute_chemical_distance(graph1 : mol_graph, graph2 : mol_graph):
    sum = 0
    for e in graph1.edges + graph2.edges:
        sum += e["edge_feature"]["weight"]      #TODO: feature
    max_mcs = None
    max = 1
    for mol1 in graph1.mols:
        for mol2 in graph2.mols:
            mcs : rdFMCS.MCSResult = rdFMCS.FindMCS([mol1,mol2])
            if mcs.numAtoms > max:
                max = mcs.numAtoms
                max_mcs = mcs
    mcs_mol = Chem.MolFromSmarts(max_mcs)
    mcs_sum = 0
    for e in mcs_mol.GetBonds():
        e : Chem.Bond = e
        mcs_sum += e.GetBondTypeAsDouble()
    return sum - 2*mcs_sum

def transform(G : mol_graph, omega : list, G_R : reaction_graph):
    def omega_inv(i):
        for j,p in enumerate(omega):
            if i == p:
                return j
        raise ValueError("Permutation is not applicable")
    phi_ = {}
    for u in G.nodes:
        for v in G.nodes:
            if u == v:
                continue
            if (omega[u],omega[v]) not in G_R.psi.keys():
                continue
            #? Aromatic Bond, Tiple Bond, Double Bond, Simple Bond, Hydrogen Bond
            print("Reaction between: 1. G_R.psi, 2. G.data_before, 3. G.data_after")
            print(u,v,G_R.psi[(omega[u],omega[v])])
            print(u,v,G.get_edge_data(u,v))
            x = -1
            if (u,v) in G.phi.keys():
                x = G.phi[(u,v)] + G_R.psi[(omega[u],omega[v])]
                G[u][v]["bond_type"] = x
            else:
                x = G_R.psi[(omega[u],omega[v])]
                G.add_edge(u,v,bond_type=x)
            phi_.update({(u,v): x}) if x >= 0.0 else ValueError("Transformation is not feasible")
            # G[u][v]["bond_type"] = G[u][v]["bond_type"] + G_R.psi[(omega[u],omega[v])]
            if x <= 0.0: # G[u][v]["bond_type"] == 0:
                G.remove_edge(u,v)
            print(u,v,G.get_edge_data(u,v))
    G.phi = phi_
    return G
