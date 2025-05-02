import os
import json 
from src.data import CSV
from src.model import CNN
from src.process import CNNTuning
from kerastuner import BayesianOptimization
from src.util import timestamp

class CNNTuningExecution:
    """Class responsible for executing the CNN tuning process."""
    
    @classmethod
    def execute(cls, data_folder: str, objective: str = 'val_loss', max_trials: int = 30):
        """Executes the CNN tuning process.

        Args:
            data_folder (str): Path to the folder containing the dataset.
            objective (str): The optimization objective for the tuner. Defaults to 'val_loss'.
            max_trials (int): The maximum number of trials for the tuner. Defaults to 30.

        Returns:
            None
        """
        path = os.path.join('build', 'tune', os.path.basename(os.path.dirname(data_folder)), timestamp())
        tuner = BayesianOptimization(CNN.hypermodel, objective, max_trials, executions_per_trial=1, directory=path, project_name='search')
        process = CNNTuning(tuner)
        process.load(data_folder)
        process.save_metadata(path)
        process.start()
        process.save(path)

    
    @classmethod
    def execute_tuning_singleOutput(self, data_folder: str, objective: str='val_loss', max_trials: int=30, train_on_scled_labels=True): #TODO Redundant zu oben, ggf. 
        path = os.path.join('build', 'tune', os.path.basename(os.path.dirname(data_folder)), timestamp())
        tuner = BayesianOptimization(CNN.hypermodel_singleOutput, objective, max_trials, executions_per_trial=1, directory=path, project_name='search')
        process = CNNTuning(tuner)
        if train_on_scled_labels:
            process.load_scaled_labels(data_folder)
        else:
            process.load(data_folder)
        process.save_metadata(path)
        print("start_singleOutput_tuning()")
        process.start_singleOutput_tuning()
        process.save(path)