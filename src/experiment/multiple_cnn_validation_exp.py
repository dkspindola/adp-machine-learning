from src.execution import CNNValidationExecution
from numpy import random
import os
import json
from pandas import DataFrame
import pandas as pd
from src.util import timestamp

class MultipleCNNValidationExperiment:
    @classmethod
    def start(cls, model_folder: str, N: int, make_validation: bool = True):
        model_folders: list[str] = os.listdir(model_folder)
        model_name = os.path.split(model_folder)[1]
        model_folders.sort(key=int, reverse=True)

        if make_validation:
            for n in range(N):
                folder = os.path.join(model_folder, model_folders[n])
                metadata_file = os.path.join(folder, 'metadata.json')
                metadata = json.load(open(metadata_file, 'r'))
                CNNValidationExecution.execute(folder, metadata['data'])

        df = DataFrame()

        validation_folders = os.listdir(f'build/validate/{model_name}')
        for id_folder in validation_folders:
            file = os.path.join(f'build/validate/{model_name}', id_folder, 'validation_results.json')
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
        os.makedirs(f'build/validate_average/{model_name}/{timestamp()}')
        averages.to_json(f'build/validate_average/{model_name}/{timestamp()}/averages.json', index=False, indent=4)
        df.to_json(f'build/validate_average/{model_name}/{timestamp()}/dataframe.json', index=False, indent=4)
        print(averages)