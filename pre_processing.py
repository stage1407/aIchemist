#embedding
#!pip install ord-schema
#!pip install wget
import ord_schema
from ord_schema import message_helpers, validations
from ord_schema.proto import dataset_pb2
from ord_schema.proto import reaction_pb2
import os
from os.path import isfile, join
import os
import shutil
import wget # type: ignore

# download ORD data
url = "https://github.com/open-reaction-database/ord-data/archive/refs/heads/main.zip"
wget.download(url, "ord-data-main.zip")

# unzip
shutil.unpack_archive("ord-data-main.zip", "ord-data-main")

# prepare datastructure
# all datasets are stored in one folder
for root, dirs, files in os.walk('ord-data-main/ord-data-main/data'):
    for file in files:
        if file[0:11] == "ord_dataset":
            path_file = os.path.join(root,file)
            shutil.copy2(path_file,'datasets')



#create list of all Datasets (onlyfiles)
path = ""
path_to_data = "datasets"
path_to_raw = "raw_data"
onlyfiles = [f for f in os.listdir(path_to_data) if isfile(join(path_to_data, f))]


def remove_provenance(data):
    stack = []
    result = []
    i = 0
    while i < len(data):
        if data[i:i+12] == "provenance {":
            stack.append("{")
            i += 12
            while stack:
                i += 1
                if data[i] == "{":
                    stack.append("{")
                elif data[i] == "}":
                    stack.pop()
            # Check if this is the last '}' in the "provenance" section
            if not stack:
                i += 1
        else:
            result.append(data[i])
            i += 1
    return "".join(result)

write_to_file = True
counter = 0
schwellenwert = 0 #für den fall das das Programm unterbrochen wird und man nicht die Zeit hat es wieder vollständig laufen zu lassen bzw. token sparen muss
wrong = []
for dataset in onlyfiles:
    if counter > schwellenwert:
        print(counter)
    pb = os.path.join(path_to_data, dataset)

    #lädt einen ganzen datensatz (das sind die zip in /data) als string
    #load a full dataset (.zip in /data) as string
    data = message_helpers.load_message(pb, dataset_pb2.Dataset)
    if write_to_file:
        data_path = os.path.join(path_to_raw, dataset[:-6])
        embedding_path = os.path.join(path, "embeddings")
        embedding_path = os.path.join(embedding_path, dataset[:-6])
        try:
            os.mkdir(data_path)
        except:
            pass
        try:
            os.mkdir(embedding_path)
        except:
            pass
    data = str(data)
    #erstellt eine liste aller reactionen die in dem Datensatz vorkommen
    #creates full list of all reactions in the dataset
    splitted = data.split("reactions {")
    for x in range(len(splitted)):
        counter += 1
        #der erste eintrag ist nur overhead vom datensatz und soll ignoriert werden
        #first entry is metadata of dataset, should be ignored
        #some entrys are unusable (wrong)
        #when programm crashed ignore all data that is already in the db (>schwellenwert)
        if x != 0 and counter not in wrong and counter > schwellenwert:
            #formatiert eine reaction so, dass sie weniger tokens brauch und schreibt das in eine Datei
            #reduce token size of reactions and write to file
            text = "reactions {" + splitted[x]
            text = remove_provenance(text)
            if write_to_file:
                raw_text = open(data_path + "/data%s.txt"%x, "w")
                raw_text.write(text)
                raw_text.close()

print("finished")
