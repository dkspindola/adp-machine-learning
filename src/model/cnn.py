from src.model.machine_learning_model import MachineLearningModel
from lib.functions_CNN_Modelle.model import build_model
import numpy as np
import keras
from keras import Optimizer
from src.data import NPY
import os
from src.data import DataType
from src.process.callback import EarlyStopOnHighValLoss

class CNN(MachineLearningModel):
    """Convolutional Neural Network (CNN) model class."""

    def __init__(self):
        """Initializes the CNN model and data attributes."""
        self.model = None
        self.data: list[NPY] = None

    @classmethod
    def from_file(cls, file: str):
        """Creates a CNN instance by loading a model from a file.

        Args:
            file (str): Path to the model file.

        Returns:
            CNN: An instance of the CNN class.
        """
        cnn = cls()
        cnn.load(file)
        return cnn

    @classmethod
    def hypermodel(cls, hp):
        """Builds a hyperparameter-tuned model.

        Args:
            hp: Hyperparameter configuration.

        Returns:
            keras.Model: A compiled Keras model.
        """
        return build_model(hp)

    def load(self, file: str):
        """Loads a Keras model from a file.

        Args:
            file (str): Path to the model file.
        """
        self.model = keras.models.load_model(file)
        print(self.model.summary())

    def save(self, file: str):
        """Saves the current model to a file.

        Args:
            file (str): Path to save the model file.
        """
        os.makedirs(os.path.dirname(file), exist_ok=True)
        self.model.save(file)
        
    def fit(self, data_file: str, optimizer: Optimizer, loss: list[str], metrics: dict[str, str]):
        """Trains the model using the provided data and parameters.

        Args:
            data_file (str): Path to the data folder.
            optimizer (Optimizer): Optimizer for training.
            loss (list[str]): List of loss functions for the model outputs.
            metrics (dict[str, str]): Dictionary of metrics for evaluation.
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.load_data(data_file)

        x_train_scaled, x_test_scaled = self.data[0].array, self.data[1].array
        y_train, y_test = self.data[2].array, self.data[3].array

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)  
        # Modell mit den besten Hyperparametern aufbauen trainieren und testen
        self.model.fit(x_train_scaled, [y_train[:, 0], y_train[:, 1], y_train[:, 2]], 
                       epochs=30, validation_data=(x_test_scaled,[y_test[:, 0], y_test[:, 1], y_test[:, 2]]),
                       callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])
        
    def validate(self, data_folder: str):
        """Validates the model on the test dataset.

        Args:
            data_folder (str): Path to the folder containing test data.

        Returns:
            dict: Evaluation results as a dictionary.
        """
        self.load_data(data_folder)

        x_test_scaled = self.data[1].array
        y_test = self.data[3].array

        y_test = np.squeeze(y_test)
        
        results = self.model.evaluate(x_test_scaled, [y_test[:, 0], y_test[:, 1], y_test[:, 2]], return_dict=True)
        return results

    def load_data(self, folder: str):
        """Loads training, testing, and validation data from a folder.

        Args:
            folder (str): Path to the folder containing data files.
        """
        self.data = []
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_TRAIN_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_TEST_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_TRAIN.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_TEST.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_VALIDATE_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_VALIDATE_SCALED.value + '.npy')))

    def soft_start(self, data_file: str, optimizer: Optimizer, loss: list[str], metrics: dict[str, str], n_unfreezed_layers: int):
        """Performs a soft start by training only a subset of layers initially.

        Args:
            data_file (str): Path to the data folder.
            optimizer (Optimizer): Optimizer for training.
            loss (list[str]): List of loss functions for the model outputs.
            metrics (dict[str, str]): Dictionary of metrics for evaluation.
            n_unfreezed_layers (int): Number of layers to unfreeze during soft start.

        Raises:
            ValueError: If `n_unfreezed_layers` is greater than the total number of layers.
        """
        # Freeze all layers except the last one
        N = len(self.model.layers)
        print("Number of Layers: ", N)
        if n_unfreezed_layers > N:
            raise ValueError("Number of unfreezed layers cannot be greater than total number of layers")
        for layer in self.model.layers[:-n_unfreezed_layers]:
            layer.trainable = False

        self.fit(data_file, optimizer, loss, metrics)

        # Unfreeze all layers after soft start
        for layer in self.model.layers:
            layer.trainable = True