#Import Bayesian Hyperparameter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers
from Load_Data_for_Modelling_Function import Data_for_Model
from Splitting_Scaling_Function import Split_Scaling
import matplotlib.pyplot as plt
import kerastuner as kt
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import shutil
import os
import json

# Import für Aktuell bestes Modell aufbauen 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import kerastuner as kt
import shutil
import os
import json
import gc

#Imports für die Validierung
import random 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from Fensterung_Scaling_DeepLearning import Fensterung_Scale
from Load_Data_for_Modelling import Get_data
from sklearn.metrics import mean_absolute_error
import pandas as pd




#Hyperparametersuche mit Bayes
def bayesian_hyperparameter_search(X_train_scaled, Y_train, X_val_scaled, Y_val, X_test_scaled, Y_test, tuner_directory, save_path):
        """
    Parameters:
    X_train_scaled : numpy array
        Scaled training input data.
    Y_train : numpy array
        Training labels.
    X_val_scaled : numpy array
        Scaled validation input data.
    Y_val : numpy array
        Validation labels.
    X_test_scaled : numpy array
        Scaled testing input data.
    Y_test : numpy array
        Testing labels.
    tuner_directory : str
        Directory to save tuner logs and checkpoints.
    save_path : str
        Directory to save the best hyperparameters and the best model.

    Returns:
    dict
        Best hyperparameters found by the tuner.
    keras.Model
        The trained model with the best hyperparameters.
    """

    # Liste leeren, um Suche zu beschleunigen
    results = list()
    # Vorherige Tuner löschen um Leistung freizugeben
    tuner_directory = 'my_dir'
    if os.path.exists(tuner_directory):
        shutil.rmtree(tuner_directory)

    # Subfunktion wird in tuner = Bayesian<Optmization aufgerufen aufgerufen 
    def build_model(hp):
        # CInputs Layer definieren (10er Window Size, 11 Features), wenn Window Size angepasst wird, hier auch anpassen
        input_layer = layers.Input(shape=(10,11))
        
        # Hyperparameter für die Anzahl an Conv Layer
        num_layers_conv = hp.Int('num_layers_conv', 0, 6)
        print(f'Anzahl an Conv Layers: {num_layers_conv}')
        
        # Hyperparameter für die Lernrate
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
        
        # Übergebe den Input Layer als x für die folgende for Schleife
        x = input_layer
        
        # Iteriere über die gewählte Anzahl an Conv Schichten
        for i in range(num_layers_conv):
            # Definition weiterer Hyperparameter innerhalb der Schleife für jede Schicht
            units_conv = hp.Int(f'units_conv{i}', min_value=32, max_value=512, step=32)
            activation_layer = hp.Choice(f'activation_conv{i}', values =['relu','tanh'])
            kernel_size = hp.Choice(f'kernel_{i}', values = [2,3,4,5])
            l2_regulizer =hp.Float(f'l2_conv{i}', min_value=0.0, max_value=0.01, step=0.001)
            print(f'Kernel Size {i} ist: {kernel_size}')
            
            # Nur die letzten 3 Pooling Schichten dürfen eine Größe größer 1 haben, damit bei der Window Size 10 und einer größeren Anzhal an Conv Layer kein Fehler auftritt
            if i >= num_layers_conv - 3:  # Letzte drei Pooling-Schichten
                pool_size = 2
            else:
                pool_size = 1

            # Conv Schicht mit Parametern
            x = layers.Conv1D(filters=units_conv, kernel_size=kernel_size, activation=activation_layer, padding='same', kernel_regularizer=keras.regularizers.l2(l2_regulizer))(x)
            #Max Pooling mit Parametern
            x = layers.MaxPooling1D(pool_size=pool_size)(x)
            
        # Flatten Schicht für Fully Connected Layer
        flatten = layers.Flatten()(x)

        # Fully Connected Part (MLP) / Dense Schichten
        
        # Definition der Anzahl an Dense Layers
        num_layers_fully = hp.Int('num_layers_fully', 0,6)
        print(f'Anzahl an Fully Connected Layers: {num_layers_fully}')

        # Übergabe der Flatten Schicht
        y = flatten
        
        for i in range(num_layers_fully):
            # Weitere Hyperparameter
            units_dense = hp.Int(f'units_dense{i}', min_value=32, max_value=512, step=32)
            activation_layer_dense = hp.Choice(f'activation_dense{i}', values =['relu','tanh'])
            l2_dense_x = hp.Float(f'l2_dense{i}', min_value=0.0, max_value=0.01, step=0.001)
            
            #Dense Schicht 
            y= layers.Dense(units_dense, activation=activation_layer_dense, kernel_regularizer=keras.regularizers.l2(l2_dense_x) )(y)
            

        # Output Layers definieren
        X_output = layers.Dense(1, activation='linear', name='Verstellweg_X')(y)
        Y_output = layers.Dense(1, activation='linear', name='Verstellweg_Y')(y)
        Phi_output = layers.Dense(1, activation='linear', name='Verstellweg_Phi')(y)

        # Liste erstellen für alle Outputs
        outputs = [X_output, Y_output, Phi_output]

        # Modell definieren 
        model = keras.Model(inputs=input_layer, outputs=outputs)

        # Kompilieren des Modells
        model.compile(optimizer=keras.optimizers.Adam(learning_rate), 
                    loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error'], 
                    metrics={'Verstellweg_X': 'mae', 'Verstellweg_Y': 'mae', 'Verstellweg_Phi': 'mae'})

        # Modell zusammenfassen
        model.summary()
        
        return model 

    # Keras Tuner initialisieren
    tuner = kt.BayesianOptimization(build_model,
                                    objective='val_loss',
                                    max_trials=30,
                                    executions_per_trial=1,
                                    directory=tuner_directory,
                                    project_name='CNN_Hyperparametertuning_BayesianOptimization')
    # Custom Callbacks: Falls Val Loss drei mal in Folge über 2.5 ist wird zur nächsten Hyperparameterkombination gesprungen
    class EarlyStopOnHighValLoss(tf.keras.callbacks.Callback):
        def __init__(self, threshold, patience=3):
            super(EarlyStopOnHighValLoss, self).__init__()
            self.threshold = threshold
            self.patience = patience
            self.wait = 0
            
        def on_epoch_end(self, epoch, logs=None):
            val_loss = logs.get('val_loss')
            if val_loss is not None and val_loss > self.threshold:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
            else:
                self.wait = 0


    # 2. Callback: Falls der Val Loss 3 mal in Folge keine Verbesserung zeigt, wird die aktuelle Suche unterbrochen und die nächsten Parameter ausgewählt
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # Aufrufen des ersten Callbacks, der zuvor definiert wurde: Grenzwert und Anzahl an Epochen in FOlge können hier festgelegt werden
    early_stop_on_high_val_loss = EarlyStopOnHighValLoss(threshold=2.5, patience=3)  

    # Hyperparametersuche durchführen mit Validationsdaten validieren
    tuner.search(X_train_scaled, [Y_train[:, 0], Y_train[:, 1], Y_train[:, 2]],
                epochs=30,
                validation_data=(X_val_scaled, [Y_val[:, 0], Y_val[:, 1], Y_val[:, 2]]),
                callbacks=[early_stopping,early_stop_on_high_val_loss])

    # Optimale Hyperparameter zurückgeben lassen
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Speichere Hyperparameter in einer JSON Datei
    best_hyperparameters = best_hps.values
    hyperparameters_pfad = os.path.join(save_path, 'best_hyperparameters.json')
    with open(hyperparameters_pfad, 'w') as json_file:
        json.dump(best_hyperparameters, json_file)

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

    # Modell mit den besten Hyperparametern aufbauen trainieren und testen
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(X_train_scaled, [Y_train[:, 0], Y_train[:, 1], Y_train[:, 2]],
                            epochs=50,
                            validation_data=(X_test_scaled,[Y_test[:, 0], Y_test[:, 1], Y_test[:, 2]]),
                            callbacks=[early_stopping])

    # Speichern des Modells
    model_pfad = os.path.join(save_path, 'best_model.h5')
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

def bestes_model(X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels):
    """
    Baut und trainiert das CNN-Modell, führt Vorhersagen durch und berechnet den Mean Absolute Error (MAE) für die Vorhersagen.
    
    Input-Parameter:
    - X_train_scaled (ndarray): Skaliertes Trainings-Feature-Set.
    - X_val_scaled (ndarray): Skaliertes Validierungs-Feature-Set.
    - X_test_scaled (ndarray): Skaliertes Test-Feature-Set.
    - Y_train_scaled (ndarray): Skaliertes Trainings-Label-Set.
    - Y_val_scaled (ndarray): Skaliertes Validierungs-Label-Set.
    - Y_test_scaled (ndarray): Skaliertes Test-Label-Set.
    - Y_train (ndarray): Trainings-Label-Set.
    - Y_val (ndarray): Validierungs-Label-Set.
    - Y_test (ndarray): Test-Label-Set.
    - scalers_features (obj): Skalierer für Features (nicht verwendet in der Funktion).
    - scaler_labels (obj): Skalierer für Labels (nicht verwendet in der Funktion).
    
    Rückgabe:
    - mae_X (float): Mean Absolute Error für Verstellweg_X.
    - mae_Y (float): Mean Absolute Error für Verstellweg_Y.
    - mae_Phi (float): Mean Absolute Error für Verstellweg_Phi.
    - df_Fehler (DataFrame): DataFrame mit den Fehlern der Vorhersagen (X, Y, Phi).
    """
     
    # TensorFlow-Sitzung zurücksetzen
    tf.keras.backend.clear_session()
    gc.collect()

    # CNN Modell definieren
    input_layer = layers.Input(shape=(10,11))

    #Struktur bestes Modell
    conv_1 = layers.Conv1D(filters=160, kernel_size=2, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(input_layer)
    pool_1 = layers.MaxPooling1D(pool_size=2)(conv_1)
    conv_2 = layers.Conv1D(filters=480, kernel_size=4, activation='relu', padding='same', strides=1, kernel_regularizer=keras.regularizers.l2(0.01))(pool_1)
    pool_2 = layers.MaxPooling1D(pool_size=2)(conv_2)

    # Flatten Schicht
    flatten = layers.Flatten()(pool_2)
  
    #FUlly Connected Schicht
    dense_layer = layers.Dense(64, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(flatten)

    # Output Layers definieren
    X_output = layers.Dense(1, activation='linear', name='Verstellweg_X')(dense_layer)
    Y_output = layers.Dense(1, activation='linear', name='Verstellweg_Y')(dense_layer)
    Phi_output = layers.Dense(1, activation='linear', name='Verstellweg_Phi')(dense_layer)

    # Liste erstellen für alle Outputs
    outputs = [X_output, Y_output, Phi_output]

    # Modell definieren 
    model = keras.Model(inputs=input_layer, outputs=outputs)
    
    # Hyperparameter 3 60 Trials bestes Modell
    model.compile(optimizer=keras.optimizers.Adam(0.0003255639325303961), 
                loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error'], 
                metrics={'Verstellweg_X': 'mae', 'Verstellweg_Y': 'mae', 'Verstellweg_Phi': 'mae'})
    
    # Modell zusammenfassen
    #model.summary()

    # Hier nur ein Callback um nach drei mal nicht verbessern des Val Loss das Trainieren zu beenden, um Overfitting zu vermeiden
    # Zweiter Callback wird nicht benötigt, weil die val_losses ja niedrig sind
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    CNN = model.fit(X_train_scaled, [Y_train[:,0], Y_train[:,1], Y_train[:,2]],
                            epochs=20,
                            validation_data=(X_val_scaled, [Y_val[:, 0], Y_val[:, 1], Y_val[:, 2]]),
                            callbacks=[early_stopping])
    
    # Vorhersagen für den Testdatensatz erstellen
    predictions = model.predict(X_test_scaled)
    X_p, Y_p, Phi_p = predictions
    
    # Bestimme die Fehler jeder einzelnen Vorhersage
    Fehler_X = Y_test[:,0]-X_p[:,0]
    Fehler_Y = Y_test[:,1]-Y_p[:,0]
    Fehler_Phi = Y_test[:,2]-Phi_p[:,0]
    
    # print(Fehler_X.shape)
    # print(Fehler_X)
      
    # Fehler in einen DataFrame konvertieren, für spätere Dichteverteilungen, Labels für tiefere Analysen
    df_Fehler = pd.DataFrame({
        'Label_X': Y_test[:,0],
        'Label_Y': Y_test[:,1],
        'Label_Phi': Y_test[:,2],
        'Fehler_X': Fehler_X,
        'Fehler_Y': Fehler_Y,
        'Fehler_Phi': Fehler_Phi})
    
    # MAEs berechnen
    mae_X = mean_absolute_error(Y_test[:, 0], X_p)
    mae_Y = mean_absolute_error(Y_test[:, 1], Y_p)
    mae_Phi = mean_absolute_error(Y_test[:, 2], Phi_p)

    # Folgendes war für den Test mit skalierten Labels, hat zu keiner Verbesserung geführt, deshalb hier nicht verwendet
    # X und Y kombinieren da diese zusammen sakliert werden
    # XY_p = np.column_stack((X_p, Y_p))

    # Rückskalierung der Vorhersagen
    # XY_pred = scaler_Y_mm.inverse_transform(XY_p)
    # X_pred, Y_pred = XY_pred[:, 0], XY_pred[:, 1]
    # Phi_pred = scaler_Y_phi.inverse_transform(Phi_p.reshape(-1, 1)).flatten()
    
    # MAE für den Testdatensatz berechnen
    # mae_X = mean_absolute_error(Y_test[:,0], X_p)
    # mae_Y = mean_absolute_error(Y_test[:,1], Y_p)
    # mae_Phi = mean_absolute_error(Y_test[:,2], Phi_p)

    # Printe die MAEs jeden Durchlaufes
    print(f"Mean Absolute Error for Verstellweg_X: {mae_X}")
    print(f"Mean Absolute Error for Verstellweg_Y: {mae_Y}")
    print(f"Mean Absolute Error for Verstellweg_Phi: {mae_Phi}")
    
    # Returne MAEs und das Dataframe der Fehler
    return mae_X, mae_Y, mae_Phi, df_Fehler

def validiere_modelle_oI(data):
    # Random Seed für die 10 fache Validation definieren
    random.seed(2)
    # Generieren einer Liste von 10 eindeutigen zufälligen Ganzzahlen zwischen 0 und 100
    Random_numbers = random.sample(range(101), 10)
    print(Random_numbers)
    # Leere Liste der Fehler zum appenden der 10er Validation
    Liste_Fehler_Blechsplit = []
    Liste_Fehler_Standardsplit= []

    # Leere Liste für die MAEs
    Liste_MAEs_Blechsplit =[]
    Liste_MAEs_Standardsplit = []

    # Leere Dataframes, falls nur eine der beiden Schleifen durchlaufen wird (für spätere Excel)
    MAE_StandardSplit_leer = pd.DataFrame(columns=['CV', 'Datentyp','Error', 'X', 'y', 'phi'])
    MAE_BlechSplit_leer = pd.DataFrame(columns=['CV', 'Datentyp','Error', 'X', 'y', 'phi'])

    # Schleife über den Standard Split
    for n in Random_numbers:
        
        # Skalierung und Fensterung der Daten
        X_train, X_val, X_test, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, Angepasste_Blechnummern_test = Fensterung_Scale(data, Validation_data=1, random=n, Train_Test_Split =1, size=0.2)
        # Berechnung der MAEs (aufrufen der obigen Funktion)
        mae_X, mae_Y, mae_phi, df_Fehler = bestes_model(X_train, X_val, X_test, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels)
        
        #Liste appenden
        MAE_StandardSplit_df = pd.DataFrame([{'CV':n, 'Datentyp': 'Standardsplit', 'Error' : 'MAE', 'X': mae_X, 'y': mae_Y, 'phi': mae_phi}])
        Liste_MAEs_Standardsplit.append(MAE_StandardSplit_df)
        
        #Fehler liste anpassen und appenden
        df_Fehler.insert(loc=0, column='SplitMethode', value='Standardsplit')
        df_Fehler.insert(loc=1, column='CV', value=n)
        Liste_Fehler_Standardsplit.append(df_Fehler)

    # Nach 10 maligem Durchlaufen zusammenfügen der Listen
    MAE_Standardsplit = pd.concat(Liste_MAEs_Standardsplit, ignore_index=True)
    Fehler_Standardsplit_df = pd.concat(Liste_Fehler_Standardsplit, ignore_index=True)

    #Schleife für den Blech SPlit
    for n in Random_numbers:
        
        X_train, X_val, X_test, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, Angepasste_Blechnummern_test   = Fensterung_Scale(data, Validation_data=1, random=n, Train_Test_Split =2, size=0.2, window_size=25)
        mae_X, mae_Y, mae_phi, df_Fehler = bestes_model(X_train, X_val, X_test, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels)
        
        MAE_BlechSplit_df = pd.DataFrame([{'CV':n, 'Datentyp': 'Blechsplit', 'Error' : 'MAE', 'X': mae_X, 'y': mae_Y, 'phi': mae_phi}])
        Liste_MAEs_Blechsplit.append(MAE_BlechSplit_df)
        
        df_Fehler.insert(loc=0, column='SplitMethode', value='Blechsplit')
        df_Fehler.insert(loc=1, column='CV', value=n)
        Liste_Fehler_Blechsplit.append(df_Fehler)
        
    MAE_Blechsplit = pd.concat(Liste_MAEs_Blechsplit, ignore_index=True)
    Fehler_Blechsplit_df = pd.concat(Liste_Fehler_Blechsplit, ignore_index=True)
        
        
    # Ausgabe der DataFrames
    # print("MAE Standard Split:")
    # print(MAE_StandardSplit_df)
    print("\nMAE Blech Split:")
    print(MAE_Blechsplit)

# Validierung des CNNs mit interpolierten Daten
# Gleiche Vorgehensweise wie oben
def validiere_modelle_mI(data)
    random.seed(2)
    # Generieren einer Liste von 10 eindeutigen zufälligen Ganzzahlen zwischen 0 und 100
    Random_numbers = random.sample(range(101), 10)
    print(Random_numbers)

    Random_numbers = Random_numbers[6:]
    print(Random_numbers)

    df_Int, Interpoliertes_df = Get_data(0,1800,1,2)

    Liste_Fehler_Blechsplit = []
    Liste_Fehler_Standardsplit= []

    Liste_MAEs_Blechsplit =[]
    Liste_MAEs_Standardsplit = []

    MAE_StandardSplit_leer = pd.DataFrame(columns=['CV', 'Datentyp','Error', 'X', 'y', 'phi'])
    MAE_BlechSplit_leer = pd.DataFrame(columns=['CV', 'Datentyp','Error', 'X', 'y', 'phi'])

    for n in Random_numbers:
        
        X_train, X_val, X_test, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels = Fensterung_Scale(df_Int, interpoliertesdf=Interpoliertes_df, Validation_data=1, random=n, Train_Test_Split =1, size=0.2, Interpolation=1)
        mae_X, mae_Y, mae_phi, df_Fehler = bestes_model(X_train, X_val, X_test, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels)
        
        MAE_StandardSplit_df = pd.DataFrame([{'CV':n, 'Datentyp': 'Standardsplit', 'Error' : 'MAE', 'X': mae_X, 'y': mae_Y, 'phi': mae_phi}])
        Liste_MAEs_Standardsplit.append(MAE_StandardSplit_df)
        
        df_Fehler.insert(loc=0, column='SplitMethode', value='Standardsplit')
        df_Fehler.insert(loc=1, column='CV', value=n)
        Liste_Fehler_Standardsplit.append(df_Fehler)
        
    MAE_Standardsplit = pd.concat(Liste_MAEs_Standardsplit, ignore_index=True)
    Fehler_Standardsplit_df = pd.concat(Liste_Fehler_Standardsplit, ignore_index=True)

    for n in Random_numbers:
        
        X_train, X_val, X_test, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, X_test_scaled_int, Y_test_interpolation, Blechnummern_Test_Int   = Fensterung_Scale(df_Int, interpoliertesdf=Interpoliertes_df, Validation_data=1, random=n, Train_Test_Split =2, size=0.2, Interpolation=1)
        mae_X, mae_Y, mae_phi, df_Fehler = bestes_model(X_train, X_val, X_test_scaled_int, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test_interpolation, scalers_features, scaler_labels)
        
        MAE_BlechSplit_df = pd.DataFrame([{'CV':n, 'Datentyp': 'Blechsplit', 'Error' : 'MAE', 'X': mae_X, 'y': mae_Y, 'phi': mae_phi}])
        Liste_MAEs_Blechsplit.append(MAE_BlechSplit_df)
        
        df_Fehler.insert(loc=0, column='SplitMethode', value='Blechsplit')
        df_Fehler.insert(loc=1, column='CV', value=n)
        Liste_Fehler_Blechsplit.append(df_Fehler)
        
    MAE_Blechsplit = pd.concat(Liste_MAEs_Blechsplit, ignore_index=True)
    Fehler_Blechsplit_df = pd.concat(Liste_Fehler_Blechsplit, ignore_index=True)
        
        
    # Ausgabe der DataFrames
    # print("MAE Standard Split:")
    # print(MAE_StandardSplit_df)
    print("\nMAE Blech Split:")
    print(MAE_Blechsplit)

#Berechnung von Statistischen Werten aus Validiierungsfunktion
def statische_Berechnungen(MAE_Standardsplit, MAE_Blechsplit, Ordner):
    ''' Parameter:
    - MAE_Standardsplit (DataFrame): Ein DataFrame, der die MAE-Werte für das Standard-Split-Verfahren enthält.
    - MAE_Blechsplit (DataFrame): Ein DataFrame, der die MAE-Werte für das Blech-Split-Verfahren enthält.
    - Ordner (str): Der Ordnerpfad, in dem die CSV-Dateien gespeichert werden sollen.

    Rückgabe:
    - None: Die Funktion speichert die Ergebnisse in CSV-Dateien, gibt jedoch keine Daten zurück.
    '''
    # Berechne Mittelwert und Std über die 10 Folds
    Mean_Standard = MAE_Standardsplit[['X','y','phi']].mean() 
    Mean_Blech = MAE_Blechsplit[['X','y','phi']].mean() 
    Std_Standard = MAE_Standardsplit[['X','y','phi']].std()
    Std_Blech = MAE_Blechsplit[['X','y','phi']].std()

    # Füge die Mittelwerte und Std den Dataframes hinzu
    MAE_StandardSplit_df = pd.concat([MAE_Standardsplit, pd.DataFrame([{'CV': 'Mittelwert', 'Datentyp': 'Standardsplit', 'Error' : 'MAE', 'X': Mean_Standard[0], 'y': Mean_Standard[1], 'phi': Mean_Standard[2]}])], ignore_index=True)
    MAE_BlechSplit_df = pd.concat([MAE_Blechsplit, pd.DataFrame([{'CV': 'Mittelwert', 'Datentyp': 'Blechsplit', 'Error' : 'MAE', 'X': Mean_Blech[0], 'y': Mean_Blech[1], 'phi': Mean_Blech[2]}])], ignore_index=True)
    MAE_StandardSplit_comp = pd.concat([MAE_StandardSplit_df, pd.DataFrame([{'CV': 'Standardabweichung', 'Datentyp': 'Standardsplit', 'Error' : 'MAE', 'X': Std_Standard[0], 'y': Std_Standard[1], 'phi': Std_Standard[2]}])], ignore_index=True)
    MAE_BlechSplit_comp = pd.concat([MAE_BlechSplit_df, pd.DataFrame([{'CV': 'Standardabweichung', 'Datentyp': 'Blechsplit', 'Error' : 'MAE', 'X': Std_Blech[0], 'y': Std_Blech[1], 'phi': Std_Blech[2]}])], ignore_index=True)

    # print(MAE_BlechSplit)
    # print(MAE_StandardSplit)

    Errors_for_CSV = pd.concat([MAE_StandardSplit_comp, MAE_BlechSplit_comp], axis=1)
    print(Errors_for_CSV['X'])

    # Columns die umgewandelt werden in String
    Errors_for_CSV.columns = ['CV', 'Datentyp', 'Error', 'X', 'y', 'phi', 'CV1',
        'Datentyp1', 'Error1', 'X1', 'y1', 'phi1']

    # Umwandlung für die Speicherung in CSVs
    for Column in Errors_for_CSV.columns:
            Errors_for_CSV[Column] = Errors_for_CSV[Column].astype(str).str.replace('.', ',')

    for Column in MAE_BlechSplit_comp.columns:
            MAE_BlechSplit_comp[Column] = MAE_BlechSplit_comp[Column].astype(str).str.replace('.', ',')
            
    for Column in Fehler_Blechsplit_df:
            Fehler_Blechsplit_df[Column] = Fehler_Blechsplit_df[Column].astype(str).str.replace('.', ',')

    # Speichern der Fehlerliste in einer CSV
    Fehler_Blechsplit_df.to_csv(f'{Ordner}\\Fehler_CNN_Standardsplit_Interpolationsfaktor2.csv', index=True, sep=';')

    # Speichern der MAEs und Std in einer CSV
    Errors_for_CSV.to_csv(f'{Ordner}\\CNN_Standardsplit_Interpolationsfaktor2[6-10].csv', index=True, sep=';')


