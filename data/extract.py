import json
from enum import Enum
import os
import periodictable
from typing import dict, float


class DatasetType(Enum):
    TRAINING = "./data/json/train"
    TEST = "./data/json/train"
    VALIDATION = "./data/json/validation"

#FileLoader
class Channel():
    def __init__(self, dataset : DatasetType):
        self.path = dataset.value

    def getMessagePaths(self):
        messages = []
        for subdir in os.listdir(self.path):
            subpath = os.path.join(self.path, subdir)
            for msg_file in os.listdir(subpath):
                msg = os.path.join(subpath, msg_file)
                messages.append(msg)
        return messages

class Message():
    def __init__(self, message_path):
        self.msg : dict = json.load(message_path)

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
        for path in ch.getMessagePaths():
            msg = Message(path)
            #! Flag solvents when finding the maximal common substructure (But should be mostly estinguished by itself)
            extracted = derive_from_data(msg)
            if extracted is not None:
                self.data.append(extracted)

def derive_from_data(msg : Message):
    #TODO: Extract Inputs, Outcomes and important conditions of the reaction (temperature, pressure, concentration/ratio, ...)
    try:
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
        
        # Inputs
        inputs = msg.getInputs()
        if not inputs:
            return None
        for _, inp_val in inputs.items():
            for comp in inp_val.get("components", []):  #?      does this work or is there another layer between?!?!
                role = comp.get("reaction_role", "").upper()
                identifiers = comp.get("identifiers", [])
                smiles = next((id["value"] for id in identifiers if id["type"] == "SMILES"), None)
                amount = comp.get("amount", {})
                if not smiles or not amount:
                    return None
                if role == "REACTANT":
                    molar_mass = periodictable.formula(smiles).mass if smiles else 1
                    mol_amount = convert_to_mol(amount, molar_mass)
                    if mol_amount is None:
                        return None
                    reaction_data["educts"].append(smiles)
                    reaction_data["educt_amounts"].append(mol_amount)
                elif role == "SOLVENT":
                    reaction_data["solvents"].append(smiles)
                elif role == "CATALYST":
                    reaction_data["catalysts"].append(smiles)

        # Conditions
        conditions = msg.getConditions()
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
            return None
        
        for outcome in outcomes:
            for product in outcome.get("products", []):
                identifiers = product.get("identifiers", [])
                smiles = next((id["value"] for id in identifiers if id["type"] == "SMILES"), None)
                amount = product.get("measurements", [{}])[0].get("amount", {})
                if not smiles or not amount:
                    return None
                molar_mass = periodictable.formula(smiles).mass if smiles else 1
                mol_amount = convert_to_mol(amount, molar_mass)
                if mol_amount is None:
                    return None
                reaction_data["products"].append(smiles)
                reaction_data["product_amounts"].append(mol_amount)
        
        return reaction_data

    except Exception as e:
        print(f"Error occured by extracting datasets: {e}")
        return None
    
def convert_to_mol(amount : dict, molar_mass : float) -> float:
    try:
        if "mass" in amount:
            mass_value = amount["mass"]["value"]
            unit = amount["mass"]["units"]
            if unit == "GRAM":
                mass_value *= 1
            elif unit == "KILOGRAM":
                mass_value *= 1_000
            elif unit == "MILLIGRAM":
                mass_value *= 0.001
            elif unit == "MICROGRAM":
                mass_value *= 0.000001
            return mass_value / molar_mass
        elif "volume" in amount:
            volume_value = amount["volume"]["value"]
            unit = amount["volume"]["units"]
            if unit == "LITER":
                volume_value *= 1
            elif unit == "MILLILITER":
                volume_value *= 0.001
            
            if "density" in amount:
                density_value = amount["density"]["value"]  # density in g/mL
                mass_value = volume_value * density_value
                return mass_value / molar_mass
            else:
                return None
        elif "moles" in amount:
            return amount["moles"]["value"]
        
        else:
            return None
        
    except Exception as e:
        print(f"Error occured by extracting datasets: {e}")
        return None


