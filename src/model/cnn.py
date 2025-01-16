from src.model.machine_learning_model import MachineLearningModel
from lib.Fensterung_Scaling_DeepLearning import Fensterung_Scale
from lib.functions_CNN_Modelle.model import build_model
import numpy as np
import keras
from keras import Optimizer
from pandas import DataFrame

class CNN(MachineLearningModel):
    FOLDER = 'build/model/cnn'

    def __init__(self):        
        self.model = None

    @classmethod
    def hypermodel(cls, hp):
        return build_model(hp)

    def load(self, file: str):
        self.model = keras.models.load_model(file)

    def save(self, file: str):
        self.model.save(file)
        
    def fit(self, data: DataFrame, optimizer: Optimizer, loss: list[str], metrics: dict[str, str]):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        x_train, _, x_test, _, _, _, y_train, _, y_test, _, _, _ = Fensterung_Scale(data, Validation_data=1, random=42, Train_Test_Split=2, window_size=10)
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)  
        # Modell mit den besten Hyperparametern aufbauen trainieren und testen
        self.model.fit(x_train, [y_train[:, 0], y_train[:, 1], y_train[:, 2]], 
                       epochs=50, validation_data=(x_test,[y_test[:, 0], y_test[:, 1], y_test[:, 2]]),
                       callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])