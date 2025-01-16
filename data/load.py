import json
from datetime import datetime
import pandas as pd # type: ignore
import random
import os
import glob
from ord_schema.message_helpers import load_message # type: ignore
from ord_schema.proto import dataset_pb2 # type: ignore
from google.protobuf.json_format import MessageToJson # type: ignore

TRAIN_RATIO = 0.5
TEST_RATIO = 0.25
VALID_RATIO = 0.25

random.seed(10)

start_time = datetime.now()

dataset_times = []

def main():
    print("Clearing split of dataset...")

    dirs = ["train","test","validation"]
    for dir in dirs:
        path = f"./data/json/{dir}"
        datasets = os.listdir(path)
        for dataset in datasets:
            files = glob.glob(f"{path}/{dataset}/*")
            for f in files:
                os.remove(f)
            os.rmdir(f"{path}/{dataset}")
        
    def dataset_split(randn):
        if randn < TRAIN_RATIO:
            return dirs[0]
        elif TRAIN_RATIO <= randn < TEST_RATIO + TRAIN_RATIO:
            return dirs[1]
        else:
            return dirs[2]

    print("Pre-Processing reaction data and split it...")

    import_path = "./data/import"
    datasets = os.listdir(import_path)
    i = 0
    for dataset in datasets:
        t_start = datetime.now()
        zips = os.listdir(f"{import_path}/{dataset}")
        j = 0
        randn = random.random()
        os.mkdir(f"./data/json/{dataset_split(randn)}/{dataset}")
        for zip in zips:
            input_fname = f"{import_path}/{dataset}/{zip}"
            output_fname = f"{import_path}/{dataset}/{str(j)}.pb.gz"
            os.rename(input_fname, output_fname)
            rxnset = load_message(
                output_fname,
                dataset_pb2.Dataset,
            )
            # take one reaction message from the dataset for example
            for rxn in rxnset.reactions:
                rxn_json = json.loads(MessageToJson(message=rxn,including_default_value_fields=False,preserving_proto_field_name=True,indent=2,sort_keys=False,use_integers_for_enums=False,descriptor_pool=None,float_precision=None,ensure_ascii=True))
                with open(f"./data/json/{dataset_split(randn)}/{dataset}/rxn{zip}.json", "w", encoding="utf-8") as f:
                    json.dump(rxn_json, f, ensure_ascii=True, indent=4)
            j += 1
        i += 1
        dataset_times.append(datetime.now() - t_start)
        print(f"Pre-Processed {i} of {len(datasets)} datasets...")

main()

print(f"Needed time was : {datetime.now()-start_time}.")
print(f"Average processing time per dataset was : {pd.to_timedelta(pd.Series(dataset_times)).mean()}.")
