import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable 

@register_keras_serializable(package="custom_layers")
class SparseStackLayer(Layer):
    """
    Mit dieser Klasse kann mein ein Modell erzeugen, das die drei Outputs zu einem Output Array verbindet.
    Dies ist notwendig, um für TradaboostR2 das Modell mit drei seperaten Outputs zu einem Modell mit einem Output zu wandeln.
    """
    def __init__(self, **kwargs):
        super(SparseStackLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Kombiniert Liste von Tensoren (z. B. [batch, 1], [batch, 1], [batch, 1])
        # zu einem Tensor mit Shape (batch, 3)
        return tf.concat(inputs, axis=-1)