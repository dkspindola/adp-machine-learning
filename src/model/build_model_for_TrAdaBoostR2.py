import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from .sparse_layer_TrAdBoostR2 import SparseStackLayer


def build_model_combinedoutputs(model_path: str, learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Konvertiert das Modell mit drei Outputs in eins mit einem Output Array. Dabei wird die sparse stack layer hinzugefügt. 
    Diese hat keine trainierbaren gewichte sondern verbindet ledigleich die Outputs in einen einen Output array.

    Args:
        model_path (str): Pfad zur .h5-Datei des gespeicherten Basis-Modells.
        learning_rate (float): Lernrate für den Optimizer.

    Returns:
        tf.keras.Model: Kompiliertes Keras-Modell mit erweitertem Output.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modellpfad nicht gefunden: {model_path}")

    base_model = load_model(model_path, compile=False)  # compile=False, falls custom layers

    # Output zusammenführen
    outputs = base_model.outputs
    combined_output = SparseStackLayer(name="sparse_output")(outputs)

    extended_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=combined_output,
        name="extended_model"
    )

    extended_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_absolute_error",
        metrics=["mae"]
    )

    return extended_model
