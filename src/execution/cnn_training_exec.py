from src.model import CNN
from src.data import CSV
from kerastuner import BayesianOptimization
import keras

class CNNTrainingExecution:
    @classmethod
    def execute(cls):
        csv=CSV.from_file('assets/data.csv', sep=';', decimal=',')
        cnn = CNN()
        cnn.load('build/tune/1737063960/best-model.h5')
        cnn.fit(csv.df, optimizer=keras.optimizers.Adam(), loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error'], metrics={'x': 'mae', 'y': 'mae', 'phi': 'mae'})
        
        