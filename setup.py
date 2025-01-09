import os

path = "./data/json"
dirs = ["base","test","validation"]
for dir in dirs:
    os.mkdir(f"{path}/{dir}")