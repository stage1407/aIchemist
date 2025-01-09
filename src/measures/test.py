from rdkit import Chem
from rdkit.Chem import rdFMCS
import random

# Generate a list of 100 random molecule SMILES strings for the example
# For a real-world scenario, you'd typically load this from a database or file
# Here we'll use simple alkane, alcohol, and amine structures for variety
alkyl_groups = ['C', 'CC', 'CCC', 'CCCC']
functional_groups = ['O', 'N']
smiles_list = []

for _ in range(100):
    alkyl = random.choice(alkyl_groups)
    func_group = random.choice(functional_groups)
    smiles_list.append(alkyl + func_group)

# Convert the SMILES strings to RDKit Mol objects
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

# Find the Maximum Common Substructure across all molecules
mcs = rdFMCS.FindMCS(mols, timeout=60)  # Set a timeout to avoid long computation times

# Get the SMARTS pattern of the MCS
mcs_smarts = mcs.smartsString

# Convert the MCS SMARTS to a Mol object
mcs_mol = Chem.MolFromSmarts(mcs_smarts)

# Print the results
print(f"Maximum Common Substructure (SMARTS): {mcs_smarts}")

# Optionally, draw the MCS
from rdkit.Chem import Draw
Draw.MolToImage(mcs_mol).show()


def main():
    #TODO: Implement a Prototyp
    pass