from src.model import CNN
from src.util import timestamp
import keras

class CNNTrainingExecution:
    @classmethod
    def execute(cls, model_file: str, data_file: str, save_filename: str, learning_rate: float):
        cnn = CNN.from_file(model_file)
        cnn.fit(data_file, optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error'], metrics={'x': 'mae', 'y': 'mae', 'phi': 'mae'})
        cnn.save(f'build/train/{timestamp()}/{save_filename}.h5')