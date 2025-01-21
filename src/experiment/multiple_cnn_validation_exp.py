from src.execution import CNNValidationExecution
from numpy import random
import os
import json
from pandas import DataFrame
import pandas as pd

class MultipleCNNValidationExperiment:
    @classmethod
    def start(cls, model_folder: str, data_folder: str, make_validation: bool = False):
        data_folders: list[str] = os.listdir(data_folder)
        model_foders: list[str] = os.listdir(model_folder)

        if make_validation:
            for data in data_folders:
                for model in model_foders:
                    try: CNNValidationExecution.execute(os.path.join(model_folder, model), os.path.join(data_folder, data))
                    except: continue

        df = DataFrame()

        validation_folders = os.listdir('build/validate')
        for id_folder in validation_folders:
            file = os.path.join('build/validate', id_folder, 'validation_results.json')
            content = json.load(open(file))
            data = content['data']
            model = content['model']
            results = content['results']

            content = {}
            content['data'] = data  
            content['model'] = model

            for key, value in results.items():
                content[key] = value

            df = df._append(content, ignore_index=True)

        numeric_columns = df.select_dtypes(include='number').columns
        averages = df[numeric_columns].mean()
        print("Averages of numeric columns:")
        print(averages)