import os 
import json
from src.model import CNN
from src.util import timestamp
import keras

class CNNSoftStartExecution:
    """Handles the execution of the CNN soft start process.

    This class provides a method to execute the soft start process for a CNN model,
    including loading the model, configuring the optimizer, defining the loss and metrics,
    and saving the resulting model and metadata.
    """

    @classmethod
    def execute(cls, model_file: str, data_file: str, learning_rate: float, n_unfreezed_layers: int):
        """Executes the soft start process for a CNN model.

        Args:
            model_file (str): Path to the CNN model file.
            data_file (str): Path to the data file used for training.
            learning_rate (float): Learning rate for the optimizer.
            n_unfreezed_layers (int): Number of layers to unfreeze during training.

        This method loads the CNN model, configures the optimizer, sets the loss and metrics,
        performs the soft start process, and saves the updated model and metadata.
        """
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