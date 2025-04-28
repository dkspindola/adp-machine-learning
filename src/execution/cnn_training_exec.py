import os 
import json
from src.model import CNN
from src.util import timestamp
import keras

class CNNTrainingExecution:
    @classmethod
    def execute(cls, model_file: str, data_file: str, save_filename: str, learning_rate: float):
        """Executes the training process for a CNN model.

        Args:
            model_file (str): Path to the file containing the CNN model definition.
            data_file (str): Path to the file containing the training data.
            save_filename (str): Filename to save the trained model.
            learning_rate (float): Learning rate for the optimizer.

        Raises:
            FileNotFoundError: If the model or data file does not exist.
            ValueError: If the training process encounters invalid parameters.
        """
        cnn = CNN.from_file(model_file)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = ['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error']
        #metrics={'x': 'mae', 'y': 'mae', 'phi': 'mae'}
        metrics={'Verstellweg_X': 'mae', 'Verstellweg_Y': 'mae', 'Verstellweg_Phi': 'mae'}
        
        cnn.fit(data_file, 
                optimizer=optimizer, 
                loss=loss, 
                metrics=metrics)

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

        folder = os.path.join('build', 'train', model_name, timestamp())

        cnn.save(os.path.join(folder, save_filename))
        json.dump(metadata, open(os.path.join(folder, 'metadata' + '.json'), 'w'), indent=4)
