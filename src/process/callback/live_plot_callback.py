from IPython import display
from tensorflow import keras
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
class LivePlotCallback(keras.callbacks.Callback):
    """
    Erstellt ein Liveplot des Trainingsfortschritts.
    Entstand im Rahmen der Domainadaptaion mit TradaboostR2 sollte aber auch wo anders mit keras models funktionieren."""
    def __init__(self):
        super().__init__()
        self.history = {"loss": [], "val_loss": []}
        #self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self.history["loss"].append(logs.get("loss", 0))
        self.history["val_loss"].append(logs.get("val_loss", 0))

    def on_train_end(self, logs=None):
        logs = logs or {}
        self.history["loss"].append(logs.get("loss", 0))
        self.history["val_loss"].append(logs.get("val_loss", 0))

        plt.figure(figsize=(6, 4))
        epochs = list(range(1, len(self.history["loss"]) + 1))

        plt.plot(epochs, self.history["loss"], label="Trainingsverlust (Loss)", color="blue", marker="o")
        plt.plot(epochs, self.history["val_loss"], label="Validierungsverlust (Loss)", color="red", linestyle="dashed", marker="o")

        plt.xlabel("Epoche")
        plt.ylabel("Loss")
        plt.title("Training vs. Validierung")
        plt.grid(True)
        plt.legend()
        display.display(plt.gcf())