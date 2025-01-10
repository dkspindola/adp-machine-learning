import numpy as np
import keras
import os
import json
import matplotlib.pyplot as plt
import kerastuner
from pandas import read_csv, DataFrame
from lib.Fensterung_Scaling_DeepLearning import Fensterung_Scale
from lib.functions_CNN_Modelle.model import build_model
from src.datacontainer import Datacontainer
from src.callback import EarlyStopOnHighValLoss


class Model:
    def train(self, data: Datacontainer):
        ...


class CNN(Model):
    def __init__(self, directory):
        self.directory = directory
        self.tuner = kerastuner.BayesianOptimization(self.hypermodel,
                                objective='val_loss',
                                max_trials=30,
                                executions_per_trial=1,
                                directory=directory,
                                project_name='CNN_Hyperparametertuning_BayesianOptimization')
    def test(filename: str):
        model = keras.models.load_model(filename)
        print(model.summary())

    def train(self, model, data: DataFrame):
        X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, Angepasste_Blechnummern_test = Fensterung_Scale(data, Validation_data=1, random=42, Train_Test_Split=2, window_size=10)
        Y_train = np.squeeze(Y_train)
        Y_test = np.squeeze(Y_test)
        Y_val =np.squeeze(Y_val)
        Y_train_scaled = np.squeeze(Y_train_scaled)
        Y_val_scaled = np.squeeze(Y_val_scaled)      
        # Modell mit den besten Hyperparametern aufbauen trainieren und testen
        return model.fit(X_train_scaled, [Y_train[:, 0], Y_train[:, 1], Y_train[:, 2]],
                                 epochs=50,
                                 validation_data=(X_test_scaled,[Y_test[:, 0], Y_test[:, 1], Y_test[:, 2]]),
                                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])


    def find(self, data: DataFrame):
        X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, Angepasste_Blechnummern_test = Fensterung_Scale(data, Validation_data=1, random=42, Train_Test_Split=2, window_size=10)
       
        Y_train = np.squeeze(Y_train)
        Y_test = np.squeeze(Y_test)
        Y_val =np.squeeze(Y_val)
        Y_train_scaled = np.squeeze(Y_train_scaled)
        Y_val_scaled = np.squeeze(Y_val_scaled)        

        self.search(X_train_scaled, Y_train, X_val_scaled, Y_val)

            # Optimale Hyperparameter zurückgeben lassen
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        # Speichere Hyperparameter in einer JSON Datei
        best_hyperparameters = best_hps.values
        hyperparameters_pfad = os.path.join(self.directory, 'best_hyperparameters_CNN_Bayesian_30Trials_Interpolation.json')
        with open(hyperparameters_pfad, 'w') as json_file:
            json.dump(best_hyperparameters, json_file, indent=4)

        # Printe alle Hyperparameters
        print("All available hyperparameters:")
        print(best_hps.values)

        # Printe die Anzhal an Layers
        print(f"Best number of convolutional layers: {best_hps.get('num_layers_conv')}")
        print(f"Best number of fully connected layers: {best_hps.get('num_layers_fully')}")

        # Printe die Hyperparameter in den einzelnen Schichten hier CONV
        for i in range(best_hps.get('num_layers_conv')):
            print(f"Best units_conv{i}: {best_hps.get(f'units_conv{i}')}")
            print(f"Best activation_conv{i}: {best_hps.get(f'activation_conv{i}')}")
            print(f"Best kernel_{i} size: {best_hps.get(f'kernel_{i}')}")

        # Hier Fully
        for i in range(best_hps.get('num_layers_fully')):
            print(f"Best units_dense{i}: {best_hps.get(f'units_dense{i}')}")
            print(f"Best activation_dense{i}: {best_hps.get(f'activation_dense{i}')}")

        best_model = self.tuner.hypermodel.build(best_hps)

        history = self.train(best_model, data)

        # Speichern des Modells
        model_pfad = os.path.join(self.directory, 'best_model_CNN_60Trials_Interpolation.h5')
        best_model.save(model_pfad)

        # Plotte die losses in der Trainings- und Testkurve
        plt.figure(figsize=(12, 8))
        plt.plot(history.history['val_Verstellweg_X_mae'], label='Validation Loss - Verstellweg_X')
        plt.plot(history.history['val_Verstellweg_Y_mae'], label='Validation Loss - Verstellweg_Y')
        plt.plot(history.history['val_Verstellweg_Phi_mae'], label='Validation Loss - Verstellweg_Phi')
        plt.title('Validation Loss for Each Output')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Loss')

        plt.legend()
        plt.show()

    def search(self, X_train_scaled, Y_train, X_val_scaled, Y_val):
        self.tuner.search(X_train_scaled, [Y_train[:, 0], Y_train[:, 1], Y_train[:, 2]], epochs=30, validation_data=(X_val_scaled, [Y_val[:, 0], Y_val[:, 1], Y_val[:, 2]]), callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3), EarlyStopOnHighValLoss(threshold=2.5, patience=3)])

    def hypermodel(self, hp):
        return build_model(hp, 10, 11)

    
def train():
    data = read_csv('assets/data.csv', sep=';', decimal=',')
    model = CNN('build/tune/cnn')
    return model.find(data)

def test(filename: str):
    data = read_csv('assets/data.csv', sep=';', decimal=',')
    cnn = CNN('build/tune/cnn')
    model = keras.models.load_model(filename)
    # Recompile the model with the same optimizer, loss, and metrics used during training
    model.compile(optimizer=keras.optimizers.Adam(), 
                loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error'], 
                metrics={'Verstellweg_X': 'mae', 'Verstellweg_Y': 'mae', 'Verstellweg_Phi': 'mae'})
    # Modell zusammenfassen
    model.summary()
    
    return cnn.train(model, data)