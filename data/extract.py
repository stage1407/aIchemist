import json
from enum import Enum
import os

class Dataset(Enum):
    TRAINING = "./data/json/train"
    TEST = "./data/json/train"
    VALIDATION = "./data/json/validation"

#FileLoader
class Channel():
    def __init__(self, dataset : Dataset):
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
