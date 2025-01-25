from src.model import CNN
from src.data import CSV
from kerastuner import BayesianOptimization
from src.util import timestamp
import keras

class CNNTrainingExecution:
    @classmethod
    def execute(cls, model_file: str, data_file: str, save_filename: str):
        cnn = CNN.from_file(model_file)
        cnn.fit(data_file, optimizer=keras.optimizers.Adam(), loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error'], metrics={'Verstellweg_X': 'mae', 'Verstellweg_Y': 'mae', 'Verstellweg_Phi': 'mae'})
        cnn.save(f'build/train/{timestamp()}/{save_filename.h5}')