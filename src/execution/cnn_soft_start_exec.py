import os 
import json
from src.model import CNN
from src.util import timestamp
import keras

class CNNSoftStartExecution:
    @classmethod
    def execute(cls, model_file: str, data_file: str, learning_rate: float, n_unfreezed_layers: int):
        cnn = CNN.from_file(model_file)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = ['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error']
        metrics={'Verstellweg_X': 'mae', 'Verstellweg_Y': 'mae', 'Verstellweg_Phi': 'mae'}
        
        cnn.soft_start(data_file, optimizer, loss, metrics, n_unfreezed_layers)

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
            "metrics": metrics,
            "n_unfreezed_layers": n_unfreezed_layers
        }

        folder = os.path.join('build', 'soft-start', model_name, timestamp())

        cnn.save(os.path.join(folder, "cnn.h5"))
        json.dump(metadata, open(os.path.join(folder, 'metadata' + '.json'), 'w'), indent=4)