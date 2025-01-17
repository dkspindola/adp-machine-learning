from src.model import CNN
from src.data import CSV
from kerastuner import BayesianOptimization
import keras

class CNNTrainingExecution:
    @classmethod
    def execute(cls, model_file: str, data_folder: str):
        cnn = CNN.from_file(model_file)
        cnn.fit(data_folder, optimizer=keras.optimizers.Adam(), loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error'], metrics={'x': 'mae', 'y': 'mae', 'phi': 'mae'})