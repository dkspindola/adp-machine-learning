from src.model.machine_learning_model import MachineLearningModel
from lib.functions_CNN_Modelle.model import build_model,build_model_output_vektor
from src.data.output_type import OutputTarget

import numpy as np
import keras
from keras import Optimizer
from src.data import NPY
import os
from src.data import DataType
from src.process.callback import EarlyStopOnHighValLoss
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Concatenate, Dense, Lambda, Layer, concatenate

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
    
    @classmethod
    def hypermodel_singleOutput(cls, hp):
        return build_model_output_vektor(hp)

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
    
    def fit_specific_output(self, data_file: str, optimizer: Optimizer, loss, metrics, selected_output: OutputTarget):#str = None):
        """
        Trains the model using the provided data and parameters.
        #TODO Redundant zu oben, erst mal debuggen dann löschen
        Args:
            data_file (str): Path to the data folder.
            optimizer (Optimizer): Optimizer for training.
            loss: Single loss function (str) or list of loss functions for multi-output.
            metrics: Dict of metrics for evaluation.
            selected_output (str, optional): If provided, only the corresponding output will be trained (e.g., "Verstellweg_X").
        """
        self.load_data(data_file)
        x_train_scaled, x_val_scaled = self.data[0].array, self.data[1].array
        y_train, y_val = np.squeeze(self.data[2].array), np.squeeze(self.data[3].array)

        # TODO ACHTUNG Reihnfolge muss stimmen!!!
        #if selected_output:
        output_index = selected_output.get_index()
        y_train = y_train[:, output_index]
        y_val = y_val[:, output_index]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model.fit(
            x_train_scaled,
            y_train if selected_output else [y_train[:, 0], y_train[:, 1], y_train[:, 2]],
            epochs=30,
            validation_data=(x_val_scaled, y_val if selected_output else [y_val[:, 0], y_val[:, 1], y_val[:, 2]]),
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
        )

    def validate_specific_output(self, data_file: str, selected_output: OutputTarget ):
        self.load_data(data_file)
        x_test_scaled= self.data[4].array
        y_test = np.squeeze(self.data[6].array)
        
        #Validieren des Modells
        results = self.model.evaluate(x_test_scaled, y_test[:,selected_output.get_index()], return_dict=True)
        return results

    def validate(self, data_folder: str):
        """Validates the model on the test dataset.

        Args:
            data_folder (str): Path to the folder containing test data.

        Returns:
            dict: Evaluation results as a dictionary.
        """
        self.load_data(data_folder)

        x_test_scaled = self.data[1].array
        y_test = self.data[3].array # TODO: Müsste das nicht index 5 sein

        y_test = np.squeeze(y_test)
        
        results = self.model.evaluate(x_test_scaled, [y_test[:, 0], y_test[:, 1], y_test[:, 2]], return_dict=True)
        return results

    def load_data(self, folder: str):
        """Loads training, testing, and validation data from a folder.

        Args:
            folder (str): Path to the folder containing data files.
        """
        self.data = []
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_TRAIN_SCALED.value + '.npy'))) # 0
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_TEST_SCALED.value + '.npy'))) # 1
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_TRAIN.value + '.npy'))) # 2
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_TEST.value + '.npy'))) # 3
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_VALIDATE_SCALED.value + '.npy'))) # 4
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_VALIDATE_SCALED.value + '.npy'))) # 5

        #ACHTUNG: Dese neu hinzugefügt, nach ADP
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_VALIDATE.value + '.npy'))) # 6

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