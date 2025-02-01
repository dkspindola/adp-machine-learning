from src.execution import CNNValidationExecution
from numpy import random
import os
import json
from pandas import DataFrame
import pandas as pd
from src.util import timestamp

class MultipleCNNValidationExperiment:
    @classmethod
    def start(cls, model_folder: str, data_folder: str, N: int, make_validation: bool = True):
        data_folders: list[str] = os.listdir(data_folder)
        model_folders: list[str] = os.listdir(model_folder)

        data_folders.sort(key=int, reverse=True)
        model_folders.sort(key=int, reverse=True)


        if make_validation:
            for n in range(N):
                CNNValidationExecution.execute(os.path.join(model_folder, model_folders[n]), os.path.join(data_folder, data_folders[n]))
                

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

        print(df)
        numeric_columns = df.select_dtypes(include='number').columns
        averages = df[numeric_columns].mean()
        os.makedirs(f'build/validate_average/{timestamp()}')
        averages.to_json(f'build/validate_average/{timestamp()}/averages.json', index=False, indent=4)
        print(averages)