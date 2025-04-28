import json 
from pandas import DataFrame
import os
import pandas as pd

def convert_to_csv(json_file: dict[str]):
    """Converts a JSON file to a CSV file.

    Args:
        json_file (dict[str]): Path to the JSON file.

    Returns:
        None. The function saves the CSV file in the same directory as the JSON file.
    """
    d = json.load(open(json_file, "r"))
    df = DataFrame.from_dict(d, orient='index')
    df.transpose
    df.to_csv(os.path.splitext(json_file)[0] + '.csv')


def read_validation_results(folder: str):
    """Reads validation results from JSON files in a folder and combines them into a DataFrame.

    Args:
        folder (str): Path to the folder containing validation result JSON files.

    Returns:
        DataFrame: A DataFrame containing combined validation results.
    """
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