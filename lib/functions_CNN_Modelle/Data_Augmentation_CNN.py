#Imports für transformerbasierten CNN Modellaufbau bzgl. DataAugmentation 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import kerastuner as kt
import shutil
import os
import json
import gc
from sklearn.metrics import mean_absolute_error

#Imports for Calculations af Basis der tranferbasierten Modellbildung
import random
import pandas as pd
from Splitting_Scaling_Function_SkalierungY import Split_Scaling
from Load_Data_for_Modelling_Function import Data_for_Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Fensterung_Scaling_CNN_ValData_Interpolation_Test import Fensterung_Scale
from sklearn.metrics import mean_absolute_error
from Load_Data_for_Modelling_Interpolation import Interpolation


#Funktion "bestes_modell_CNN" und "bestes_modell_Transformer" werden von Caclculation-FUnktion aufgerufen
#transformerbasierten CNN Modellaufbau bzgl. DataAugmentation 
def bestes_model_CNN(X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, X_test_scaled_int=None, Y_test_interpolation=None, Blechnummern_Test_Int=None, n=0, Int=1 ):
      
    Y_train = np.squeeze(Y_train)
    Y_test = np.squeeze(Y_test)
    Y_val =np.squeeze(Y_val)
    Y_train_scaled = np.squeeze(Y_train_scaled)
    Y_val_scaled = np.squeeze(Y_val_scaled)
    
    if Int ==1:
        Y_test_interpolation = np.squeeze(Y_test_interpolation)
        
        print(Y_test_interpolation.shape)
        print(X_test_scaled_int.shape)
    
    # TensorFlow-Sitzung zurücksetzen
    tf.keras.backend.clear_session()
    gc.collect()

# CNN Modell definieren
    input_layer = layers.Input(shape=(10,11))

    # # Convolutional Layers
    # conv_1 = layers.Conv1D(filters=320, kernel_size=3, activation='relu', strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.008))(input_layer)
    # pool_1 = layers.MaxPooling1D(pool_size=1)(conv_1)
    # conv_2 = layers.Conv1D(filters=288, kernel_size=5, activation='relu', padding='same', strides=1, kernel_regularizer=keras.regularizers.l2(0.004))(pool_1)
    # pool_2 = layers.MaxPooling1D(pool_size=2)(conv_2)
    # conv_3 = layers.Conv1D(filters=128, kernel_size=4, activation='relu', padding='same', strides=1, kernel_regularizer=keras.regularizers.l2(0.002))(pool_2)
    # pool_3 = layers.MaxPooling1D(pool_size=2)(conv_3)
    # conv_4 = layers.Conv1D(filters=416, kernel_size=5, activation='relu', padding='same', strides=1, kernel_regularizer=keras.regularizers.l2(0.01))(pool_3)
    # pool_4 = layers.MaxPooling1D(pool_size=2)(conv_4)
    
    conv_1 = layers.Conv1D(filters=160, kernel_size=2, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.01))(input_layer)
    pool_1 = layers.MaxPooling1D(pool_size=2)(conv_1)
    conv_2 = layers.Conv1D(filters=480, kernel_size=4, activation='relu', padding='same', strides=1, kernel_regularizer=keras.regularizers.l2(0.01))(pool_1)
    pool_2 = layers.MaxPooling1D(pool_size=2)(conv_2)

    flatten = layers.Flatten()(pool_2)

    # print(flatten)
    # print(type(flatten))
    # print(flatten.shape)

    # # Fully Connected Layers
    # dense_layer = layers.Dense(448, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.001))(flatten)
    # dense_layer_1 = layers.Dense(96, activation='relu', kernel_regularizer=keras.regularizers.l2(0.003))(dense_layer)
    # dense_layer_2 = layers.Dense(448, activation='relu', kernel_regularizer=keras.regularizers.l2(0.008))(dense_layer_1)
    # dense_layer_3 = layers.Dense(160, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.003))(dense_layer_2)
    # dense_layer_4 = layers.Dense(320, activation='relu', kernel_regularizer=keras.regularizers.l2(0.007))(dense_layer_3)
    
     #Hyperparametersuche 3
    dense_layer = layers.Dense(64, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(flatten)
    

    # Output Layers definieren
    X_output = layers.Dense(1, activation='linear', name='Verstellweg_X')(dense_layer)
    Y_output = layers.Dense(1, activation='linear', name='Verstellweg_Y')(dense_layer)
    Phi_output = layers.Dense(1, activation='linear', name='Verstellweg_Phi')(dense_layer)

    # Liste erstellen für alle Outputs
    outputs = [X_output, Y_output, Phi_output]

    # Modell definieren 
    model = keras.Model(inputs=input_layer, outputs=outputs)

    # # Kompilieren des Modells
    # model.compile(optimizer=keras.optimizers.Adam(0.00013453713072694953), 
    #             loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error'], 
    #             metrics={'Verstellweg_X': 'mae', 'Verstellweg_Y': 'mae', 'Verstellweg_Phi': 'mae'})
    
        # Kompilieren des Modells
    model.compile(optimizer=keras.optimizers.Adam(0.0003255639325303961), 
                loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error'], 
                metrics={'Verstellweg_X': 'mae', 'Verstellweg_Y': 'mae', 'Verstellweg_Phi': 'mae'})


    # Modell zusammenfassen
    #model.summary()

    # Define early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    CNN = model.fit(X_train_scaled, [Y_train[:,0], Y_train[:,1], Y_train[:,2]],
                            epochs=30,
                            validation_data=(X_val_scaled, [Y_val[:, 0], Y_val[:, 1], Y_val[:, 2]]),
                            callbacks=[early_stopping])
    if Int==1:
        # Vorhersagen für den Testdatensatz erstellen
        predictions_alle = model.predict(X_test_scaled)
        X_p_alle, Y_p_alle, Phi_p_alle = predictions_alle
        
        predictions_Versuche = model.predict(X_test_scaled_int)
        X_p_versuch, Y_p_versuch, Phi_p_versuch = predictions_Versuche

        #print(predictions_Versuche.shape)
        print(predictions_Versuche)
        print(X_p_versuch.shape)
        print(X_p_versuch)
        print(Y_test_interpolation[:,0].shape)
        print(Y_test_interpolation[:,0])
        
            #print(predictions)
        mae_X_all = mean_absolute_error(Y_test[:, 0], X_p_alle)
        mae_Y_all = mean_absolute_error(Y_test[:, 1], Y_p_alle)
        mae_Phi_all = mean_absolute_error(Y_test[:, 2], Phi_p_alle)
        
        mae_X_versuch = mean_absolute_error(Y_test_interpolation[:, 0], X_p_versuch)
        mae_Y_versuch = mean_absolute_error(Y_test_interpolation[:, 1], Y_p_versuch)
        mae_Phi_versuch = mean_absolute_error(Y_test_interpolation[:, 2], Phi_p_versuch)
        
            # Fehler in einen DataFrame konvertieren
        MAEs_CNN = pd.DataFrame({
            'Interpolationsfaktor': [n],
            'MAE_X_AlleDaten': [mae_X_all],
            'MAE_Y_AlleDaten': [mae_Y_all],
            'MAE_Phi_AlleDaten': [mae_Phi_all],
            'MAE_X_Versuchsdaten': [mae_X_versuch],
            'MAE_Y_Versuchsdaten': [mae_Y_versuch],
            'MAE_Phi_Versuchsdaten': [mae_Phi_versuch]})

        print(f"Dataframe für MAEs CNN: {MAEs_CNN}")
        
    else:
        # Vorhersagen für den Testdatensatz erstellen
        predictions_alle = model.predict(X_test_scaled)
        X_p_alle, Y_p_alle, Phi_p_alle = predictions_alle
        
        print(X_p_alle.shape)
        
        mae_X_all = mean_absolute_error(Y_test[:, 0], X_p_alle)
        mae_Y_all = mean_absolute_error(Y_test[:, 1], Y_p_alle)
        mae_Phi_all = mean_absolute_error(Y_test[:, 2], Phi_p_alle)
        
        print(mae_X_all.shape)
        print(mae_X_all)
        print(n)
        
            # Fehler in einen DataFrame konvertieren
        MAEs_CNN = pd.DataFrame({
            'Interpolationsfaktor': [n],
            'MAE_X_AlleDaten': [mae_X_all],
            'MAE_Y_AlleDaten': [mae_Y_all],
            'MAE_Phi_AlleDaten': [mae_Phi_all],
            'MAE_X_Versuchsdaten': ['N/A'],
            'MAE_Y_Versuchsdaten': ['N/A'],
            'MAE_Phi_Versuchsdaten': ['N/A']})
        
    
    return MAEs_CNN

def bestes_model_Transformer(X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, X_test_scaled_int=None, Y_test_interpolation=None, Blechnummern=None, n=0, Int=1 ):
      
    Y_train = np.squeeze(Y_train)
    Y_test = np.squeeze(Y_test)
    Y_val =np.squeeze(Y_val)
    Y_train_scaled = np.squeeze(Y_train_scaled)
    Y_val_scaled = np.squeeze(Y_val_scaled)
    
    if Int==1:
        Y_test_interpolation = np.squeeze(Y_test_interpolation)
        
        print(Y_test_interpolation.shape)
        print(X_test_scaled_int.shape)
    
    
    
    # TensorFlow-Sitzung zurücksetzen
    tf.keras.backend.clear_session()
    gc.collect()
     
    # Transformer Encoder Block
    def transformer_encoder(inputs, head_size, num_heads, ff_dim_1, ff_dim_2,  dropout_1, dropout_2, kernel_size1, kernel_size2):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads)(x,x)
        x = layers.Dropout(dropout_1)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim_1, kernel_size=kernel_size1, activation="relu",padding='same')(x)
        x = layers.Dropout(dropout_2)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, padding='same')(x)
        
        return x + res


    input_layer = layers.Input(shape=(10, 11))
        
    x = input_layer

    x = transformer_encoder(x, 250, 4, 288, 224, 0, 0.3, 3, 5)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    # Fully Connected Layers

        # Fully Connected Layers
    dense_layer = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.002))(x)
    dense_layer_1 = layers.Dense(448, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(dense_layer)
    dense_layer_2 = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0))(dense_layer_1)
    dense_layer_3 = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0))(dense_layer_2)

    # Output Layers
    X_output = layers.Dense(1, activation='linear', name='Verstellweg_X')(dense_layer_3)
    Y_output = layers.Dense(1, activation='linear', name='Verstellweg_Y')(dense_layer_3)
    Phi_output = layers.Dense(1, activation='linear', name='Verstellweg_Phi')(dense_layer_3)

    outputs = [X_output, Y_output, Phi_output]

    model = keras.Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(0.0003269360363083118), 
                    loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error'], 
                    metrics={'Verstellweg_X': 'mae', 'Verstellweg_Y': 'mae', 'Verstellweg_Phi': 'mae'})

    model.summary()
    
    # Define early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    Transformer_model = model.fit(X_train_scaled, [Y_train[:,0], Y_train[:,1], Y_train[:,2]],
                            epochs=30,
                            validation_data=(X_val_scaled, [Y_val[:, 0], Y_val[:, 1], Y_val[:, 2]]),
                            callbacks=[early_stopping])
    if Int==1:
            
        # Vorhersagen für den Testdatensatz erstellen
        predictions_alle = model.predict(X_test_scaled)
        X_p_alle, Y_p_alle, Phi_p_alle = predictions_alle
        
        predictions_Versuche = model.predict(X_test_scaled_int)
        X_p_versuch, Y_p_versuch, Phi_p_versuch = predictions_Versuche

        #print(X_train_flat.shape)
        #print(X_val_flat.shape)
            #print(predictions)
        mae_X_all = mean_absolute_error(Y_test[:, 0], X_p_alle)
        mae_Y_all = mean_absolute_error(Y_test[:, 1], Y_p_alle)
        mae_Phi_all = mean_absolute_error(Y_test[:, 2], Phi_p_alle)
        
        mae_X_versuch = mean_absolute_error(Y_test_interpolation[:, 0], X_p_versuch)
        mae_Y_versuch = mean_absolute_error(Y_test_interpolation[:, 1], Y_p_versuch)
        mae_Phi_versuch = mean_absolute_error(Y_test_interpolation[:, 2], Phi_p_versuch)
        
            # Fehler in einen DataFrame konvertieren
        MAEs_Transformer = pd.DataFrame({
            'Interpolationsfaktor': [n],
            'MAE_X_AlleDaten': [mae_X_all],
            'MAE_Y_AlleDaten': [mae_Y_all],
            'MAE_Phi_AlleDaten': [mae_Phi_all],
            'MAE_X_Versuchsdaten': [mae_X_versuch],
            'MAE_Y_Versuchsdaten': [mae_Y_versuch],
            'MAE_Phi_Versuchsdaten': [mae_Phi_versuch]})

    else:
           # Vorhersagen für den Testdatensatz erstellen
        predictions_alle = model.predict(X_test_scaled)
        X_p_alle, Y_p_alle, Phi_p_alle = predictions_alle
        
        mae_X_all = mean_absolute_error(Y_test[:, 0], X_p_alle)
        mae_Y_all = mean_absolute_error(Y_test[:, 1], Y_p_alle)
        mae_Phi_all = mean_absolute_error(Y_test[:, 2], Phi_p_alle)
        
            # Fehler in einen DataFrame konvertieren
        MAEs_Transformer = pd.DataFrame({
            'Interpolationsfaktor': [n],
            'MAE_X_AlleDaten': [mae_X_all],
            'MAE_Y_AlleDaten': [mae_Y_all],
            'MAE_Phi_AlleDaten': [mae_Phi_all],
            'MAE_X_Versuchsdaten': ['N/A'],
            'MAE_Y_Versuchsdaten': ['N/A'],
            'MAE_Phi_Versuchsdaten': ['N/A']})
        
        
        
    return MAEs_Transformer



def calculate_for_interpolation(Random_split=11, num_loops=4):
    """
    Berechnet den Mean Absolute Error (MAE) für Interpolation und speichert die Ergebnisse
    für CNN- und Transformer-Modelle.

    Parameters:
    -------------
    random_split : int, optional
        Zufallszahl zur Steuerung der Datenaufteilung. Standardwert: 11.

    num_loops : int, optional
        Anzahl der Iterationen zur Berechnung der MAE. Standardwert: 4.

    Returns:
    -------------
    tuple: Enthält zwei DataFrames mit MAE-Werten für CNN- und Transformer-Modelle.
        - MAEs_Interpolation_CNN_df
        - MAEs_Interpolation_Transformer_df
    """

    # random.seed(1)
    # Random_numbers =  [random.randint(0, 100) for _ in range(10)]
    # print(Random_numbers)


    # MAE_StandardSplit_df = pd.DataFrame(columns=['CV', 'Datentyp','Error', 'X', 'y', 'phi'])
    # MAE_BlechSplit_df = pd.DataFrame(columns=['CV', 'Datentyp','Error', 'X', 'y', 'phi'])

    #data = Data_for_Model(0,1800)

    # Initialisierung der Variablen für Fehlerdaten
    MAEs_Interpolation_CNN = []
    MAEs_Interpolation_Transformer = []
   
    for n in range(num_loops):
        
        if n < 1:
            print(f'Im Loop der if Abfrage_{n}')
            data = Data_for_Model(0,1800)
            X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, Angepasste_Blechnummern_test = Fensterung_Scale(data, Validation_data=1, random=Random_Split, Train_Test_Split=2, window_size=10)
            MAEs_CNN = bestes_model_CNN(X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, n=n, Int=0 )
            MAEs_trans = bestes_model_Transformer(X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, n=n, Int=0 )
            MAEs_CNN.insert(loc=0, column='CV', value = Random_Split)
            MAEs_trans.insert(loc=0, column='CV', value = Random_Split)
            MAEs_CNN.insert(loc=0, column='Modell', value = 'CNN')
            MAEs_trans.insert(loc=0, column='Modell', value = 'Transformer')
            MAEs_Interpolation_CNN.append(MAEs_CNN)
            MAEs_Interpolation_Transformer.append(MAEs_trans)
            
        else:
            print(n)
            df_Int, Interpoliertes_df = Interpolation(0,1800,n)
            X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, X_test_scaled_int, Y_test_interpolation, Blechnummern_Test_Int = Fensterung_Scale(df_Int, interpoliertesdf= Interpoliertes_df,Validation_data=1, random=Random_Split, Train_Test_Split =2, size=0.2, Interpolation=1)
            MAEs_CNN = bestes_model_CNN(X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, X_test_scaled_int, Y_test_interpolation, Blechnummern_Test_Int, n=n )
            MAEs_trans = bestes_model_Transformer(X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, X_test_scaled_int=X_test_scaled_int, Y_test_interpolation=Y_test_interpolation, n=n )
            MAEs_CNN.insert(loc=0, column='CV', value = Random_Split)
            MAEs_trans.insert(loc=0, column='CV', value = Random_Split)
            MAEs_CNN.insert(loc=0, column='Modell', value = 'CNN')
            MAEs_trans.insert(loc=0, column='Modell', value = 'Transformer')
            MAEs_Interpolation_CNN.append(MAEs_CNN)
            MAEs_Interpolation_Transformer.append(MAEs_trans)
        
    MAEs_Interpolation_CNN_df = pd.concat(MAEs_Interpolation_CNN, axis=0)
    MAEs_Interpolation_Transformer_df =pd.concat(MAEs_Interpolation_Transformer, axis=0)
        
        
    # Ausgabe der DataFrames
    print("MAE CNN Interpolation:")
    print(MAEs_Interpolation_CNN_df)
    print("\nMAE Transformer Interpolation:")
    print(MAEs_Interpolation_Transformer_df)

def save_calculations(MAEs_Interpolation_CNN_df, MAEs_Interpolation_Transformer_df, output_folder, output_filename):
    # Zusammenfügen der DataFrames
    Mae_interpolation_ges = pd.concat([MAEs_Interpolation_CNN_df, MAEs_Interpolation_Transformer_df], axis=0, ignore_index=True)

    # Ersetzen von Punkten durch Kommata in allen Spalten
    for column in Mae_interpolation_ges.columns:
        Mae_interpolation_ges[column] = Mae_interpolation_ges[column].astype(str).str.replace('.', ',')

    # Speichern als CSV-Datei
    output_path = f"{output_folder}/{output_filename}"
    Mae_interpolation_ges.to_csv(output_path, index=True, sep=';')

    print(f"Datei wurde erfolgreich gespeichert unter: {output_path}")

