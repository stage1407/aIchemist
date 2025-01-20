import json
from enum import Enum
import os

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
            educt_mol_graph, product_mol_graph = derive_from_data(msg)

def derive_from_data(msg : Message):
    #TODO
    pass
