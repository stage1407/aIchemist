import json
from enum import Enum
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFreeSASA
from scipy.constants import N_A #as AVOGADRO
from math import gcd

REALISM = True

# Wrong, but limited to our resources:
AVOGADRO = 10_000

class DatasetType(Enum):
    TRAINING = "./data/json/train"
    TEST = "./data/json/test"
    VALIDATION = "./data/json/validation"

#FileLoader
class Channel():
    def __init__(self, dataset : DatasetType):
        self.path = dataset.value
        self.ld = 0

    def getMessagePaths(self):
        messages = []
        loaded = 0
        for subdir in os.listdir(self.path):
            subpath = os.path.join(self.path, subdir)
            for msg_file in os.listdir(subpath):
                loaded += 1
                msg = os.path.join(subpath, msg_file)
                messages.append(msg)
        self.ld = loaded
        return messages

class Message():
    def __init__(self, message_path : str):
        with open(message_path, 'r') as f:
            self.msg : dict = json.load(f)

    def getInputs(self):
        return self.msg.get("inputs")

    def getSetup(self):
        return self.msg.get("setup")

    def getConditions(self):
        return self.msg.get("conditions")

    def getWorkups(self):
        return self.msg.get("workups")

    def getOutcomes(self):
        return self.msg.get("outcomes")

#    def getProvenance(self):
#        return self.msg.get("provenance")


#TODO: Data Pipeline, missing gap between data/json and models/training.py
#TODO: 1. Extract Inputs (Educts + Reagents + Catalysts) and Outcomes (Products + Catalysts)
#TODO: 2. Initialize Mol_Graphs for Educts and Products AND DON'T FORGET THE REACTION GRAPHS
class Extractor():
    def __init__(self, type : DatasetType):
        ch = Channel(type)
        self.data = []
        for path in ch.getMessagePaths():
            msg = Message(path)
            # print(path)
            #! Flag solvents when finding the maximal common substructure (But should be mostly estinguished by itself)
            extracted = derive_from_data(msg)
            if extracted is not None:
                if extracted["educts"] != [] \
                    and extracted["educt_amounts"] \
                        and extracted["products"] != [] \
                        and extracted["product_amounts"] != []:
                    self.data.append(extracted)
        self.loaded_files = ch.ld

    def __len__(self):
        return len(self.data)

def derive_from_data(msg : Message):
    reaction_data = {
        "educts": [],
        "educt_amounts": [],
        "products": [],
        "product_amounts" : [],
        "solvents": [],
        "solvent_amounts": [],
        "catalysts": [],
        "catalyst_amount": [],
        "conditions": {
            "temperature": None,
            "pressure": None,
            "stirring": None
        }
    }
    try:        
        # Inputs
        inputs = msg.getInputs()
        # print(inputs)
        if not inputs: #Error check
            pass
        else:
            for _, inp_val in inputs.items():
                for comp in inp_val.get("components", []):  #?      does this work or is there another layer between?!?!
                    role = comp.get("reaction_role", "").upper()
                    identifiers = comp.get("identifiers", [])
                    smiles = next((id["value"] for id in identifiers if id["type"] == "SMILES"), None)
                    #print(smiles)
                    amount = comp.get("amount", {})
                    #print(amount)
                    if smiles is None or amount is None: #Error check if data is complete
                        pass
                    else: 
                        if role == "REACTANT":
                            mol = Chem.MolFromSmiles(smiles)
                            molar_mass = Descriptors.MolWt(mol) if mol else None
                            mol_amount = convert_to_mol(amount, molar_mass, smiles)
                            if mol_amount is None:
                                pass
                            else:
                                #print(smiles)
                                reaction_data["educts"].append(smiles)
                                reaction_data["educt_amounts"].append(mol_amount)
                        elif role == "SOLVENT":
                            #! Difficult
                            mol = Chem.MolFromSmiles(smiles)
                            molar_mass = Descriptors.MolWt(mol) if mol else None
                            mol_amount = convert_to_mol(amount, molar_mass, smiles)
                            #print(mol_amount)
                            if mol_amount is None:
                                pass
                            else:
                                reaction_data["educts"].append(smiles)
                                reaction_data["educt_amounts"].append(mol_amount)
                                reaction_data["products"].append(smiles)
                                reaction_data["product_amounts"].append(mol_amount)
                        elif role == "CATALYST":
                            #reaction_data["educts"].append(smiles)
                            #reaction_data["educt_amounts"].append(1)
                            #reaction_data["products"].append(smiles)
                            #reaction_data["product_amounts"].append(0.9)
                            pass

                        #print(amount)
        #print(reaction_data if reaction_data is not None else "") 

        # Conditions
        conditions = msg.getConditions()
        # print("Cond",conditions)
        temperature = conditions.get("temperature", {}).get("value", None)
        temp_unit = conditions.get("temperatur",{}).get("setpoint",{}).get("units", None)
        if temperature and temp_unit == "CELSIUS":
            temperature += 273.15       # Kelvin
        reaction_data["conditions"]["temperature"] = temperature

        reaction_data["conditions"]["pressure"] = conditions.get("pressure", {}).get("value", None)
        reaction_data["conditions"]["stirring"] = conditions.get("stirring", {}).get("details", None)

        # Outcomes
        outcomes = msg.getOutcomes()
        if not outcomes:
            pass
        else:
            for outcome in outcomes:
                for product in outcome.get("products", []):
                    identifiers = product.get("identifiers", [])
                    smiles = next((id["value"] for id in identifiers if id["type"] == "SMILES"), None)
                    amount = product.get("measurements", [{}])[0].get("amount", {})
                    if not smiles or not amount:
                        return None
                    mol = Chem.MolFromSmiles(smiles)
                    molar_mass = Descriptors.MolWt(mol) if mol else None
                    mol_amount = convert_to_mol(amount, molar_mass, smiles)
                    # print(mol_amount)
                    if molar_mass is None or mol_amount is None:
                        return None
                    reaction_data["products"].append(smiles)
                    reaction_data["product_amounts"].append(mol_amount)
        # print("ReactData", reaction_data)
        #print(reaction_data["educt_amounts"],reaction_data["product_amounts"])
        
        # Down-Scale Number of molecules
        print(reaction_data["educt_amounts"])
        downscale_factor = max(1,gcd(*(reaction_data["educt_amounts"] + reaction_data["product_amounts"])))
        downscale = lambda y : [max(1,x // downscale_factor) for x in y]
        reaction_data["educt_amounts"] = downscale(reaction_data["educt_amounts"])
        reaction_data["product_amounts"] = downscale(reaction_data["product_amounts"])
        print(reaction_data["product_amounts"])
    except Exception as e:
        # print(traceback.format_exc())
        # print(f"Error occured by extracting datasets: {e}")
        #print(reaction_data["educt_amounts"])
        pass
    return reaction_data

    
"""
def derive_from_data(msg):
    ""
    Extracts structured reaction data from an ORD message.
    Skips invalid reactions (i.e., missing educts, products, or amounts).

    Parameters:
    - msg (dict): A reaction message in ORD format.

    Returns:
    - dict: Processed reaction data OR None if the message is invalid.
    ""

    # Initialize reaction data structure
    reaction_data = {
        "educts": [], "educt_amounts": [],
        "products": [], "product_amounts": [],
        "solvents": [], "solvent_amounts": [],
        "conditions": {"temperature": None, "pressure": None, "stirring": None}
    }

    try:
        # **Step 1: Extract Inputs (Educts, Solvents)**
        inputs = msg.get("inputs", {})  # Retrieve all input components
        if not inputs:
            return None  # **Skip message if no inputs are present**

        for key, inp_val in inputs.items():
            role = inp_val.get("reaction_role", "").upper()
            for comp in inp_val.get("components", []):
                identifiers = comp.get("identifiers", [])
                smiles = next((id["value"] for id in identifiers if id["type"] == "SMILES"), None)
                amount = comp.get("amount", {})

                if smiles is None or not amount:
                    continue  # Skip invalid molecules

                mol = Chem.MolFromSmiles(smiles)
                molar_mass = Descriptors.MolWt(mol) if mol else None
                mol_amount = convert_to_mol(amount, molar_mass, smiles)
                if mol_amount is None:
                    continue  # Skip if amount conversion fails

                # **Educts & Solvents**
                if role == "REACTANT":
                    reaction_data["educts"].append(smiles)
                    reaction_data["educt_amounts"].append(mol_amount)
                elif role == "SOLVENT":
                    reaction_data["educts"].append(smiles)
                    reaction_data["educt_amounts"].append(mol_amount)
                    reaction_data["products"].append(smiles)
                    reaction_data["product_amounts"].append(mol_amount)

        # **Step 2: Extract Outcomes (Products)**
        outcomes = msg.get("outcomes", [])
        if not outcomes:
            return None  # **Skip message if no product info is present**

        for outcome in outcomes:
            for product in outcome.get("products", []):
                identifiers = product.get("identifiers", [])
                smiles = next((id["value"] for id in identifiers if id["type"] == "SMILES"), None)
                amount = product.get("measurements", [{}])[0].get("amount", {})

                if smiles is None or not amount:
                    continue  # Skip invalid products

                mol = Chem.MolFromSmiles(smiles)
                molar_mass = Descriptors.MolWt(mol) if mol else None
                mol_amount = convert_to_mol(amount, molar_mass, smiles)
                if mol_amount is None:
                    continue

                reaction_data["products"].append(smiles)
                reaction_data["product_amounts"].append(mol_amount)

        # **Step 3: Ensure Essential Data Exists**
        if not reaction_data["educts"] or not reaction_data["products"]:
            return None  # **Skip invalid reactions**

        # **Step 4: Extract Conditions**
        conditions = msg.get("conditions", {})
        reaction_data["conditions"]["temperature"] = (
            conditions.get("temperature", {}).get("setpoint", {}).get("value", None)
        )
        reaction_data["conditions"]["pressure"] = conditions.get("pressure", {}).get("value", None)
        reaction_data["conditions"]["stirring"] = conditions.get("stirring", {}).get("details", None)

        # **Step 5: Downscale Number of Molecules**
        all_amounts = reaction_data["educt_amounts"] + reaction_data["product_amounts"]
        if all(x == 0 for x in all_amounts):
            return None  # **Skip reaction if all amounts are zero**

        downscale_factor = max(1, gcd(*all_amounts))  # Compute GCD to simplify proportions
        downscale = lambda y: [max(1, x // downscale_factor) for x in y]
        reaction_data["educt_amounts"] = downscale(reaction_data["educt_amounts"])
        reaction_data["product_amounts"] = downscale(reaction_data["product_amounts"])

    except Exception as e:
        print(f"⚠ Error in derive_from_data: {e}")
        return None  # **Skip reactions that failed parsing**

    return reaction_data

"""

    
def convert_to_mol(amount : dict, molar_mass : float, smiles : str = None) -> float:
    def estimate_density() -> float:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol_weight = Descriptors.MolWt(mol)
            radii = rdFreeSASA.classifyAtoms(mol)
            mol_volume = rdFreeSASA.CalcSASA(mol, radii)

            estimated_volume = mol_volume * 1.5     # Approximate conversion factor

            density = mol_weight /estimated_volume if estimated_volume > 0 else None
            return round(density, 3) if density else None
        except Exception as e:
            # print(f"Error estimating density for {smiles}: {e}")
            return None
    res = 0
    try:
        if "mass" in amount:
            mass_value = amount["mass"]["value"]
            unit = amount["mass"]["units"]
            if unit == "GRAM":
                mass_value *= 1
            elif unit == "KILOGRAM":
                mass_value *= 1_000
            elif unit == "MILLIGRAM":
                mass_value *= 0.00_1
            elif unit == "MICROGRAM":
                mass_value *= 0.00_000_1
            elif unit == "NANOGRAM":
                mass_value *= 0.00_000_000_1
            res = mass_value / molar_mass

        elif "volume" in amount:
            volume_value = amount["volume"]["value"]
            unit = amount["volume"]["units"]
            if unit == "LITER":
                volume_value *= 1
            elif unit == "MILLILITER":
                volume_value *= 0.00_1
            elif unit == "MICROLITER":
                volume_value *= 0.00_000_1
            elif unit == "NANOLITER":
                volume_value *= 0.00_000_000_1

            density_value = amount.get("density", {}).get("value", None)
            if density_value is None:
                density_value = estimate_density(smiles)
            
            if density_value is not None:
                mass_value = volume_value * density_value
                res = mass_value / molar_mass
            else:
                return None

        elif "moles" in amount:
            unit = amount["volume"]["units"]
            if unit == "MOLE":
                volume_value *= 1
            elif unit == "MILLIMOLE":
                volume_value *= 0.00_1
            elif unit == "NANOMOLE":
                volume_value *= 0.00_000_1
            res = amount["moles"]["value"]
        else:
            return None
        return res*AVOGADRO if REALISM else res
        
    except Exception as e:
        # print(f"Error occured by converting dataset to mols: {e}")
        return None

for v in [DatasetType.TEST, DatasetType.TRAINING, DatasetType.VALIDATION]:
    ex = Extractor(v)
    print(str(v)+ ":", str(len(ex.data)) + "/" + str(ex.loaded_files))
    #print("Data:", ex.data)