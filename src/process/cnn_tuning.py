from src.process.process import Process
from src.serialize import Serializable
from src.data import NPY, DataType
from src.process.callback import EarlyStopOnHighValLoss, EarlyStopping

import os
import json
import numpy as np
from kerastuner import Tuner, HyperParameters

class CNNTuning(Process,Serializable):
    def __init__(self, tuner: Tuner):
        self.tuner = tuner
        self.data: list[NPY] = None # [x_train, x_validate, y_train, y_validate]
        self.datafolder = None
        self.epochs = None
        self.hyperparameters: list[HyperParameters] = []
        
    def start(self):
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

    def load(self, folder: str):
        self.datafolder = folder
        self.data = []
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_TRAIN_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.X_VALIDATE_SCALED.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_TRAIN.value + '.npy')))
        self.data.append(NPY.from_file(os.path.join(folder, DataType.Y_VALIDATE.value + '.npy')))

    def save(self, folder: str):
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
        metadata = {
            "data": self.datafolder, 
            "tuner": self.tuner.__class__.__name__,
            "executions_per_trial": str(self.tuner.executions_per_trial),
            "objective": str(self.tuner.oracle.objective),
            "max_trials": str(self.tuner.oracle.max_trials),
            "search": self.tuner.oracle.get_space().get_config()
        }
        json.dump(metadata, open(os.path.join(folder, 'metadata' + '.json'), 'w'), indent=4)