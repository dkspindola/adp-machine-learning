from src.model.machine_learning_model import MachineLearningModel
from lib.functions_CNN_Modelle.model import bulid_model_one_output

import numpy as np
import keras
from keras import Optimizer
from src.data import NPY
import os
from src.serialize import Serializable
from src.data import DataType, OutputTarget
from src.process.callback import EarlyStopOnHighValLoss
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import  Concatenate, Dense, Lambda, Layer, concatenate

class CNN3Models(Serializable):
    def __init__(self):
        """
        Diese Klasse erzeugt die drei modelle, der einzelnen Ausgänge.
        """
        # Die drei einzelnen Modelle
        self.model_X=None
        self.model_Y=None
        self.model_Phi=None

        # List dieser Modelle 
        self.models=[self.model_X,self.model_Y,self.model_Phi]

    def load_data(self, folder: str):
        """Loads training, testing, and validation data from a folder.

        Args:
            folder (str): Path to the folder containing data files.
        """
        #Input Data, immer skaliert ansonsten Unsinn
        self.x_train_scaled = NPY.from_file(os.path.join(folder, DataType.X_TRAIN_SCALED.value + '.npy'))
        self.x_test_scaled = NPY.from_file(os.path.join(folder, DataType.X_TEST_SCALED.value + '.npy'))
        self.x_validate_scaled = NPY.from_file(os.path.join(folder, DataType.X_VALIDATE_SCALED.value + '.npy'))
        
        #Original targets
        self.y_train = NPY.from_file(os.path.join(folder, DataType.Y_TRAIN.value + '.npy'))
        self.y_test = NPY.from_file(os.path.join(folder, DataType.Y_TEST.value + '.npy'))
        self.y_validate=NPY.from_file(os.path.join(folder, DataType.Y_VALIDATE.value + '.npy'))
        
        #Sclaierte Targets
        self.y_train_scaled=NPY.from_file(os.path.join(folder, DataType.Y_TRAIN_SCALED.value + '.npy'))
        self.y_test_scaled = NPY.from_file(os.path.join(folder, DataType.Y_TEST_SCALED.value + '.npy'))
        self.y_validate_scaled=NPY.from_file(os.path.join(folder, DataType.Y_VALIDATE_SCALED.value + '.npy'))

    def load(self, file):
        """TODO Noch implementieren, weiß nicht ob notwendig."""
        pass

    def save(self, file):
        """TODO Muss noch implementiert werden."""
        pass

    
    def fit_all(self, data_file: str, optimizer: Optimizer, loss: list[str], metrics: dict[str, str]):
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

    @classmethod    
    def hypermodel(cls,hp, output_idf: OutputTarget):
        return bulid_model_one_output(hp=hp,output_idf=output_idf)
