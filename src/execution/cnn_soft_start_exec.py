import os 
import json
from src.model import CNN
from src.util import timestamp
import keras

class CNNSoftStartExecution:
    @classmethod
    def execute(cls, model_file: str, data_file: str, save_filename: str, learning_rate: float):
        cnn = CNN.from_file(model_file)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = ['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error']
        metrics={'Verstellweg_X': 'mae', 'Verstellweg_Y': 'mae', 'Verstellweg_Phi': 'mae'}
        
        cnn.soft_start(data_file, optimizer, loss, metrics)

        _, model_name = os.path.split(model_file)
        model_name, _ = os.path.splitext(model_name)

        metadata = {
            "model": model_file,
            "data": data_file,
            "optimizer": {
                "name": optimizer.name,
                "learning_rate": learning_rate
            },
            "loss": loss,
            "metrics": metrics
        }

        folder = os.path.join('build', 'soft-start', model_name, timestamp())

        cnn.save(os.path.join(folder, save_filename))
        json.dump(metadata, open(os.path.join(folder, 'metadata' + '.json'), 'w'), indent=4)
        
        print(os.path.join(folder, save_filename))
