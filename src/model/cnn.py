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

    def __init__(self):        
        self.model = None
        self.data: list[NPY] = None

    @classmethod
    def from_file(cls, file: str):
        cnn = cls()
        cnn.load(file)
        return cnn

    @classmethod
    def hypermodel(cls, hp):
        return build_model(hp)

    def load(self, file: str):
        self.model = keras.models.load_model(file)

    def save(self, file: str):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        self.model.save(file)
        
    def fit(self, data_file: str, optimizer: Optimizer, loss: list[str], metrics: dict[str, str]):
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
        self.load_data(data_folder)

        x_test_scaled = self.data[1].array
        y_test = self.data[3].array

        y_test = np.squeeze(y_test)
        
        results = self.model.evaluate(x_test_scaled, [y_test[:, 0], y_test[:, 1], y_test[:, 2]], return_dict=True)
        return results

    def load_data(self, folder: str):
        self.data = []
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_TRAIN_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_TEST_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_TRAIN.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_TEST.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_VALIDATE_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_VALIDATE_SCALED.value + '.npy')))