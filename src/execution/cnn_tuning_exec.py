import os
import json 
from src.data import CSV
from src.model import CNN
from src.process import CNNTuning
from kerastuner import BayesianOptimization
from src.util import timestamp

class CNNTuningExecution:
    @classmethod
    def execute(cls, data_folder: str, objective: str='val_loss', max_trials: int=30):
        path = os.path.join('build', 'tune', os.path.basename(os.path.dirname(data_folder)), timestamp())
        tuner = BayesianOptimization(CNN.hypermodel, objective, max_trials, executions_per_trial=1, directory=path, project_name='search')
        process = CNNTuning(tuner)
        process.load(data_folder)
        process.save_metadata(path)
        process.start()
        process.save(path)