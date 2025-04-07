from src.model.cnn import CNN
from src.execution import CNNSoftStartExecution
from keras.optimizers import Adam
import json
import os

class SoftStartExperiment():
    @classmethod
    def run(cls, model_file: str, data_folder: str, lr_factors: list[float], unfreezed_layers: list[int]):
        results = []
        model_folder = os.path.dirname(model_file)
        learning_rate = None
        #load learning rate from metadata.json file
        with open(model_folder + '/metadata.json') as f:
            metadata = json.load(f)
            learning_rate = metadata['optimizer']['learning_rate']
            print(f"Learning rate from metadata: {learning_rate}")

        for factor in lr_factors:
            for n in unfreezed_layers:
                lr = factor * learning_rate
                print(f"Testing with learning rate: {lr}, unfreezed layers: {n}")
                CNNSoftStartExecution.execute(model_file, data_folder, lr, n)


