from src.model.cnn import CNN
from src.execution import CNNSoftStartExecution
from keras.optimizers import Adam
import json
import os

class SoftStartExperiment():
    """Class to perform soft start experiments with a CNN model."""

    @classmethod
    def run(cls, model_file: str, data_folder: str, lr_factors: list[float], unfreezed_layers: list[int]):
        """Runs the soft start experiment.

        This method adjusts the learning rate by multiplying it with factors
        and tests the model with different numbers of unfreezed layers.

        Args:
            model_file (str): Path to the pre-trained model file.
            data_folder (str): Path to the folder containing the dataset.
            lr_factors (list[float]): List of factors to adjust the learning rate.
            unfreezed_layers (list[int]): List of numbers of layers to unfreeze.

        Returns:
            None
        """
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


