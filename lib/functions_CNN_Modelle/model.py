import keras
from keras import layers
import tensorflow as tf
from src.data import OutputTarget
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Concatenate, Dense, Lambda, Layer, concatenate


@register_keras_serializable(package="custom_layers")
class SparseStackLayer(Layer):
    def __init__(self, **kwargs):
        super(SparseStackLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Kombiniert Liste von Tensoren (z. B. [batch, 1], [batch, 1], [batch, 1])
        # zu einem Tensor mit Shape (batch, 3)
        return tf.concat(inputs, axis=-1)


# Keras Modell definieren
def build_model(hp, window_size=10, n_features=11):
    
    # CInputs Layer definieren (10er Window Size, 11 Features), wenn Window Size angepasst wird, hier auch anpassen
    input_layer = layers.Input(shape=(window_size,n_features))
    
    # Hyperparameter für die Anzahl an Conv Layer
    num_layers_conv = hp.Int('num_layers_conv', 1, 6)
    print(f'Anzahl an Conv Layers: {num_layers_conv}')
    
    # Hyperparameter für die Lernrate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    
    # Übergebe den Input Layer als x für die folgende for Schleife
    x = input_layer
    
    # Iteriere über die gewählte Anzahl an Conv Schichten
    for i in range(num_layers_conv):
        # Definition weiterer Hyperparameter innerhalb der Schleife für jede Schicht
        units_conv = hp.Int(f'units_conv{i}', min_value=1, max_value=15, step=1)
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
    num_layers_fully = hp.Int('num_layers_fully', 1,6, 1)
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


def build_model_output_vektor(hp, window_size=10, n_features=11):
    
    # CInputs Layer definieren (10er Window Size, 11 Features), wenn Window Size angepasst wird, hier auch anpassen
    input_layer = layers.Input(shape=(window_size,n_features))
    
    # Hyperparameter für die Anzahl an Conv Layer
    num_layers_conv = hp.Int('num_layers_conv', 1, 6)
    print(f'Anzahl an Conv Layers: {num_layers_conv}')
    
    # Hyperparameter für die Lernrate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    
    # Übergebe den Input Layer als x für die folgende for Schleife
    x = input_layer
    
    # Iteriere über die gewählte Anzahl an Conv Schichten
    for i in range(num_layers_conv):
        # Definition weiterer Hyperparameter innerhalb der Schleife für jede Schicht
        units_conv = hp.Int(f'units_conv{i}', min_value=1, max_value=15, step=1)
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
    num_layers_fully = hp.Int('num_layers_fully', 1,6, 1)
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
        
    output_vector = layers.Dense(3, activation='linear', name='Verstellweg')(y)

    # Modell definieren 
    model = keras.Model(inputs=input_layer, outputs=output_vector)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
              loss='mean_absolute_error',
              metrics=['mae'])
    model.summary()
    return model 

def bulid_model_one_output(hp,output_idf : OutputTarget, window_size=10, n_features=11):
    """
    Erstellt ein Keras-Modell mit nur einem Output.
    Args:
        hp: Hyperparameter-Objekt (z.B. von Keras Tuner)
        output_idf: Zu verwendende Metrik für das Modell, eines der folgenden: 'Verstellweg_X', 'Verstellweg_Y', 'Verstellweg_Phi'
        window_size: Größe des Input-Fensters (Standard: 10)
        n_features: Anzahl der Input-Features (Standard: 11)
    Returns:
        model: Kompiliertes Keras-Modell
    """
    # Cinputs Layer definieren (10er Window Size, 11 Features), wenn Window Size angepasst wird, hier auch anpassen
    input_layer = layers.Input(shape=(window_size,n_features))
    
    # Hyperparameter für die Anzahl an Conv Layer
    num_layers_conv = hp.Int('num_layers_conv', 1, 6)
    print(f'Anzahl an Conv Layers: {num_layers_conv}')
    
    # Hyperparameter für die Lernrate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    
    # Übergebe den Input Layer als x für die folgende for Schleife
    x = input_layer
    
    # Iteriere über die gewählte Anzahl an Conv Schichten
    for i in range(num_layers_conv):
        # Definition weiterer Hyperparameter innerhalb der Schleife für jede Schicht
        units_conv = hp.Int(f'units_conv{i}', min_value=1, max_value=15, step=1)
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
    num_layers_fully = hp.Int('num_layers_fully', 1,6, 1)
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
        

    # Output Layers definieren, ACHTUNG Name!!!
    single_output = layers.Dense(1, activation='linear', name=output_idf.get_output_name())(y)
    
    # Modell definieren 
    model = keras.Model(inputs=input_layer, outputs=single_output)

    # Kompilieren des Modells
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), 
                loss='mean_absolute_error',#output_idf.get_loss_dict(),#{output_idf.value :'mean_absolute_error'}, 
                metrics=['mae']#output_idf.get_loss_metric_dict()# {output_idf.value : 'mae'})
    )
    # Modell zusammenfassen
    #model.summary()

    print("=====================================================================")
    print("Nach dem Modellbau")
    print("Output-Namen:", model.output_names)
    print("Loss dict:", model.loss)
    print("Metrics dict:", model.metrics_names)

    return model 
    