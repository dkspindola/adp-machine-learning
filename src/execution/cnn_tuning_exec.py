import os
from src.data import CSV
from src.model import CNN
from src.process import CNNTuning
from kerastuner import BayesianOptimization
from src.util import timestamp

class CNNTuningExecution:
    @classmethod
    def execute(cls, data_folder: str, objective: str='val_loss', max_trials: int=30, ts: int=None):
        path = os.path.join('build', 'tune', timestamp() if ts is None else str(ts))
        tuner = BayesianOptimization(CNN.hypermodel, objective, max_trials, executions_per_trial=1, directory=path, project_name='search')

        process = CNNTuning(tuner)
        process.load(data_folder)
        process.start()
        process.save(path)

    