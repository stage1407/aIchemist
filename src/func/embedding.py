from rdkit import Chem
import json
from datastructure import _derive_isometries

MAX_DEPTH = 10
quantum_mechanics = False       # set True for modeling pi-electrons        #!Dummy for now
geometric_properties = False    # set True for modeling hydrogen bonds      #!Dummy for now
properties = [
    "Hybridization",
    "Formal charge",
    "Implicit hydrogens",
    "Aromaticity",
    "Chirality",
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

def get_atom_properties(atom : Chem.Atom, property):
    node_properties = properties["node_features"]
    hybrid_num = {v: k for k,v in Chem.rdchem.HybridizationType.values.items()}
    chiral_num = {v: k for k,v in Chem.rdchem.ChiralType.values.items()}
    property_dict = {
        node_properties[0]: hybrid_num[atom.GetHybridization()],  # Returns the hybridization type (e.g., SP2).    #? Enhance to types
        node_properties[1]: atom.GetFormalCharge(),  # Returns the formal charge on the atom.
        node_properties[2]: atom.GetNumImplicitHs(),  # Returns the number of implicit hydrogens.
        node_properties[3]: atom.GetIsAromatic(),  # Returns True if the atom is aromatic.
        #*Note: Same as Stereochemistry 
        node_properties[4]: chiral_num[atom.GetChiralTag()],  # Returns chirality (e.g., R/S).             #? Enhance to types     
        node_properties[5]: atom.GetExplicitValence(),  # Returns the explicit valence of the atom. 
        node_properties[6]: atom.GetImplicitValence(),  # Returns the implicit valence of the atom.
        node_properties[7]: (atom.GetTotalNumHs() + atom.GetImplicitValence() + atom.GetExplicitValence() - atom.GetTotalValence()) // 2, # Calculated manually as lone pairs are not directly accessible.
        node_properties[8]: [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()],  # Returns bond types connected to the atom.
        node_properties[9]: ((atom.GetTotalNumHs() + atom.GetImplicitValence() + atom.GetExplicitValence() - atom.GetTotalValence()) // 2) == 0,  # True if no lone pairs.
        node_properties[10]: atom.GetDegree(),  # Returns the number of directly bonded atoms
        node_properties[11]: atom.GetTotalDegree(),  # Returns total degree including implicit hydrogens.
        node_properties[12]: atom.GetFormalCharge(),  # Same as Formal charge.
        node_properties[13]: (atom.GetTotalNumHs() + atom.GetImplicitValence() + atom.GetExplicitValence() - atom.GetTotalValence()) // 2,  # Same as Lone pairs.
        node_properties[14]: atom.GetIsAromatic(),  # Same as Aromaticity.
        node_properties[15]: atom.GetAtomicNum(),  # Approximate electronegativity by atomic number.
        node_properties[16]: chiral_num[atom.GetChiralTag()] if geometric_properties else _derive_isometries(),  # Same as Chirality/Stereochemistry.   #? Enhance to types
        node_properties[17]: atom.GetAtomicNum() in [6, 7, 8, 15, 16, 17],  # An approximate way to infer hydrophobicity.
    }
    return property_dict[property]

def get_bond_properties(bond : Chem.Bond, p):
    stereo_num = {v : k for k, v in Chem.rdchem.BondStereo.values.items()}
    properties = {
        "Bond_Order": bond.GetBondTypeAsDouble(),  # Returns the bond order (e.g., 1.0, 2.0).
        "Electrophilic_Character": int(bond.GetBeginAtom().GetAtomicNum() in [6, 7] and bond.GetEndAtom().GetAtomicNum() in [8, 16]),  # Approximate electrophilic character.
        "Nucleophilic_Character": int(bond.GetBeginAtom().GetAtomicNum() in [15, 16] or bond.GetEndAtom().GetAtomicNum() in [15, 16]),  # Approximate nucleophilic character.
        #"Length": rdMolTransforms.GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),  # Bond length from 3D coordinates.
        "Aromaticity": int(bond.GetIsAromatic()),  # Checks if the bond is aromatic.
        #"Orientation": "Not directly available",  # Orientation calculation requires vector math.
        "Stereo_State": stereo_num[bond.GetStereo()],  # Returns the stereochemistry of the bond.           #? Enhance to bondType?
        #"Distance": rdMolTransforms.GetBondLength(mol.GetConformer(), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())  # Same as Length.
    }
    return properties[p]
