import os 
import json
from typing import List, Optional
from src.model import CNN
from src.data import NPY, DataType, OutputTarget
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

    @classmethod
    def execute_three_models_training(cls, model_folder: str, data_file: str,
                                      save_filename="best-model",learning_rates : Optional[List[float]] = None,
                                      save_folder: str = None):
        """
        Trainiert drei getrennte Modelle (X, Y, Phi), basierend auf den gespeicherten best-model.h5
        und speichert sie samt Metadaten in getrennte Unterordnern.
        ACHTUNG: Die Learningrate wird nun aus der zugehörigen Doku gelesen
        Args:
            model_folder (str): Pfad zum Ordner, in dem die Unterordner für die drei Modelle liegen (z.B. .../Verstellweg_X/)
            data_file (str): Pfad zur Trainingsdatei (enthält alle drei Targets).
            save_filename (str): Dateiname für das Modell (.h5).
            learning_rate (float): Lernrate für den Optimierer.
        """

        timestamp_str = timestamp()
        if learning_rates is not None and not len(learning_rates)==3:
            raise ValueError("Die learningrates muss entwerde none sien, dann wird jeweils die optimale aus dem tuning genommen, oder es müssen drie Learningrates angegeben werden.")
        
        for output_target in OutputTarget:
            output_name = output_target.get_output_name()
            model_path = os.path.join(model_folder, output_name, 'best-model.h5')

            if not os.path.exists(model_path):
                print(f"Modell in {output_name} nicht gefunden: {model_path}")
                continue

            print(f"Training startet für: {output_name} basierend auf dem Best Model aus: \n{model_path}")
            
            #Einlesen der besten Learningrate:
            if learning_rates is None:
                hyperparam_path = os.path.join(model_folder, output_name, "best-hyperparameters.json")
                with open(hyperparam_path, 'r') as f:
                    hyperparams = json.load(f)
                learning_rate = hyperparams["learning_rate"]

            cnn = CNN.from_file(model_path)

            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

            # Nur den spezifischen Output verwenden
            loss = output_target.get_loss_dict()
            metrics = output_target.get_loss_metric_dict()

            cnn.fit_specific_output(data_file, optimizer=optimizer, loss=loss, metrics=metrics, selected_output=output_target)
            
            if save_folder:
                model_save_folder = os.path.join(save_folder,output_name)
            else:
                model_save_folder = os.path.join('build', 'train_single_models', timestamp_str, output_name)
            os.makedirs(model_save_folder, exist_ok=True)

            model_save_path = os.path.join(model_save_folder, f"{save_filename}.h5")
            cnn.save(model_save_path) # TODO Kommt hier das .h5 automatisch?

            metadata = {
                "model": model_path,
                "data": data_file,
                "optimizer": {
                    "name": optimizer.name,
                    "learning_rate": learning_rate
                },
                "loss": loss,
                "metrics": metrics,
                "trained_output": output_name
            }

            #Evaluieren des Modells, ansonsten das später machen müssen
            eval_data=cnn.validate_specific_output(data_file=data_file,selected_output=output_target)
            metadata["evaluation"] = eval_data

            metadata_path = os.path.join(model_save_folder, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            print(f"{output_name} Modell gespeichert in: {save_folder}")