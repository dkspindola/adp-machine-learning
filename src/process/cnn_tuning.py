from src.process.process import Process
from src.serialize import Serializable
from src.data import NPY, DataType, OutputTarget
from src.process.callback import EarlyStopOnHighValLoss, EarlyStopping

import os
import json
import numpy as np
import keras 
from kerastuner import Tuner, HyperParameters

class CNNTuning(Process,Serializable):
    """
    Class for tuning a CNN model using Keras Tuner.
    
    Attributes:
        tuner (Tuner): The Keras Tuner instance used for hyperparameter tuning.
        data (list[NPY]): List containing training and validation data.
        datafolder (str): Path to the folder containing the data.
        epochs (int): Number of epochs for training.
        hyperparameters (list[HyperParameters]): List of best hyperparameters found during tuning.
    """
    def __init__(self, tuner: Tuner):
        self.tuner = tuner
        self.data: list[NPY] = None # [x_train, x_validate, y_train, y_validate]
        self.datafolder = None
        self.epochs = None
        self.hyperparameters: list[HyperParameters] = []
        
    def start(self):
        """
        Start the tuning process by loading data and performing hyperparameter search.
        """
        x_train, x_validate = self.data[0].array, self.data[1].array
        y_train, y_validate = self.data[2].array, self.data[3].array

        y_train = np.squeeze(y_train)
        y_validate =np.squeeze(y_validate)
        
        #y_train, y_validate = np.squeeze(y_train), np.squeeze(y_validate)

        self.tuner.search(x_train, [y_train[:, 0], y_train[:, 1], y_train[:, 2]],
                          epochs=30, 
                          validation_data=(x_validate, [y_validate[:, 0], y_validate[:, 1], y_validate[:, 2]]), 
                          callbacks=[EarlyStopping(monitor='val_loss', patience=3), 
                                     EarlyStopOnHighValLoss(threshold=2.5, patience=3)],
                          verbose=1)

        # Optimale Hyperparameter zurückgeben lassen
        self.hyperparameters = self.tuner.get_best_hyperparameters(num_trials=1)

    def start_singleOutput_tuning(self):
        x_train, x_validate = self.data[0].array, self.data[1].array
        y_train, y_validate = self.data[2].array, self.data[3].array

        y_train = np.squeeze(y_train)
        y_validate =np.squeeze(y_validate)
        print(y_validate.shape)
        #y_train, y_validate = np.squeeze(y_train), np.squeeze(y_validate)

        self.tuner.search(x_train, y_train,
                          epochs=30, 
                          validation_data=(x_validate, y_validate), 
                          callbacks=[EarlyStopping(monitor='val_loss', patience=3), 
                                     EarlyStopOnHighValLoss(threshold=2.5, patience=3)],
                          verbose=1)

        # Optimale Hyperparameter zurückgeben lassen
        self.hyperparameters = self.tuner.get_best_hyperparameters(num_trials=1)
    
    def start_three_models_tuning(self, output_name : OutputTarget):
        x_train, x_validate = self.data[0].array, self.data[1].array
        y_train, y_validate = self.data[2].array, self.data[3].array

        y_train = np.squeeze(y_train)
        y_validate =np.squeeze(y_validate)
        self.tuned_models: dict[OutputTarget, keras.Model] = {}
        #Daten auswählen für das einzelne Modell
        index=output_name.get_index()
        y_train_single = y_train[:, index]
        y_val_single = y_validate[:, index]

        #Trennen der Modelle
        self.tuner.search(x_train,
                          y_train_single,#{output_name.get_output_name(): y_train_single},#
                            epochs=3, # TODO: ZUM DEBUGGEN VON 30 AUF 3 REDUZIERT ==================================================================== 
                             validation_data=(x_validate, y_val_single),#validation_data=(x_validate, {output_name.value : y_val_single}), 
                            callbacks=[EarlyStopping(
                                monitor=output_name.get_objective(),
                                patience=3,mode='min'), 
                                    EarlyStopOnHighValLoss(threshold=2.5, patience=3)],
                        verbose=1)

        # Optimale Hyperparameter zurückgeben lassen
        best_hp = self.tuner.get_best_hyperparameters(num_trials=1)
        # Hyperparameter ablegen mit Modellzuordnung
        self.tuned_models[output_name] = best_hp

    def load(self, folder: str):
        """
        Load the data from the specified folder.
        
        Args:
            folder (str): Path to the folder containing the data files.
        """
        self.datafolder = folder
        self.data = []
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_TRAIN_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_VALIDATE_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_TRAIN.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_VALIDATE.value + '.npy')))


    def load_scaled_labels(self, folder: str):
        self.datafolder = folder
        self.data = []
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_TRAIN_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_VALIDATE_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_TRAIN_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_VALIDATE_SCALED.value + '.npy')))

        #Weil kein Bock die gnaze Pipeline zu überarbeiten, hier noch ein Jason um Sonderfallt Training auf gescalten labeln zu dokumentieren
        metadata = {
            "Lable Type": "Label sind Scaliert"
        }
        json.dump(metadata, open(os.path.join(folder, 'metadata_lableType' + '.json'), 'w'), indent=4)


    def save_single_model(self, folder: str):
        """
        Save best hyperparameters and models for each OutputTarget to the specified folder.

        Args:
            folder (str): Path to the folder where everything will be saved.
        """
        os.makedirs(folder, exist_ok=True)

        if not self.tuned_models:
            print("Warnung: Keine getunten Modelle zum Speichern vorhanden.")
            return

        for output_target, best_hp in self.tuned_models.items():
            # 1. Speicherpfade definieren
            model_name = output_target.get_output_name()
            model_folder = os.path.join(folder, model_name)
            os.makedirs(model_folder, exist_ok=True)

            # 2. Hyperparameter als JSON speichern
            hp_file = os.path.join(model_folder, 'best-hyperparameters.json')
            save_best_hp=best_hp[0].values
            with open(hp_file, 'w') as json_file:
                json.dump(save_best_hp, json_file, indent=4)

            # 3. Modell mit besten Hyperparametern bauen & speichern
            best_model = self.tuner.hypermodel.build(best_hp[0])
            model_file = os.path.join(model_folder, 'best-model.h5')
            best_model.save(model_file)

            print(f"\n Modell für   {model_name} gespeichert:")
            print(f"- Hyperparameter: {hp_file}")
            print(f"- Modell:         {model_file}")


    def save(self, folder: str):
        """
        Save the best hyperparameters and model to the specified folder.

        Args:
            folder (str): Path to the folder where the hyperparameters and model will be saved.
        """
        # Speichere Hyperparameter in einer JSON Datei
        best = self.hyperparameters[0]
        best_hyperparameters = best.values
        file = os.path.join(folder, 'best-hyperparameters.json')
        with open(file, 'w') as json_file:
            json.dump(best_hyperparameters, json_file, indent=4)

        # Printe alle Hyperparameters
        print("All available hyperparameters:")
        print(best.values)

        # Printe die Anzhal an Layers
        print(f"Best number of convolutional layers: {best.get('num_layers_conv')}")
        print(f"Best number of fully connected layers: {best.get('num_layers_fully')}")

        # Printe die Hyperparameter in den einzelnen Schichten hier CONV
        for i in range(best.get('num_layers_conv')):
            print(f"Best units_conv{i}: {best.get(f'units_conv{i}')}")
            print(f"Best activation_conv{i}: {best.get(f'activation_conv{i}')}")
            print(f"Best kernel_{i} size: {best.get(f'kernel_{i}')}")

        # Hier Fully
        for i in range(best.get('num_layers_fully')):
            print(f"Best units_dense{i}: {best.get(f'units_dense{i}')}")
            print(f"Best activation_dense{i}: {best.get(f'activation_dense{i}')}")

        
        best_model = self.tuner.hypermodel.build(best)
        model_pfad = os.path.join(folder, 'best-model.h5')
        best_model.save(model_pfad)

    def save_metadata(self, folder: str):
        """
        Save metadata about the tuning process to a JSON file.

        Args:
            folder (str): Path to the folder where the metadata will be saved.
        """
        metadata = {
            "data": self.datafolder, 
            "tuner": self.tuner.__class__.__name__,
            "executions_per_trial": str(self.tuner.executions_per_trial),
            "objective": str(self.tuner.oracle.objective),
            "max_trials": str(self.tuner.oracle.max_trials),
            "search": self.tuner.oracle.get_space().get_config()
        }
        json.dump(metadata, open(os.path.join(folder, 'metadata' + '.json'), 'w'), indent=4)