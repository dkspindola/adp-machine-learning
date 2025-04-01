import json 
from pandas import DataFrame
import os
import pandas as pd

def convert_to_csv(json_file: dict[str]):
    d = json.load(open(json_file, "r"))
    df = DataFrame.from_dict(d, orient='index')
    df.transpose
    df.to_csv(os.path.splitext(json_file)[0] + '.csv')


def read_validation_results(folder: str):
    df = DataFrame()
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file == "validation_results.json":
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    folder_name = os.path.basename(root)  # Get the folder name
                    temp_df = DataFrame.from_dict([data["results"]])
                    model_file: str = data["model"]
                    temp_df.index = [model_file.split("/")[-2]]  # Set folder name as index
                    df = pd.concat([df, temp_df], ignore_index=False)
    return df