
import os
import json
import time
import numpy as np
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras.callbacks import Callback,EarlyStopping, CSVLogger 
from typing import Optional
from src.data.output_type import OutputTarget
from src.model.build_model_for_TrAdaBoostR2 import build_model_combinedoutputs, build_model_single_output 
from src.model.sparse_layer_TrAdBoostR2 import SparseStackLayer
from src.process.callback.live_plot_callback import LivePlotCallback
from adapt.instance_based import TrAdaBoostR2
from scikeras.wrappers import KerasRegressor


class TrAdaBoostR2ModelSetup:
    """
    Mit der Klasse wird das TradaboostR2 Training ausgeführt.
    Es wird eine Grundeinstellung gewählt. 
    -> Earlystopping von Tradaboost, True, False
    -> Earlystopping von Modelltraining: True, False
    -> Parameter des Trainings:
        -> 
    """
    def __init__(self,
                 model_path, # Dateipfad zum Modell, sollte .h5 Datei sein
                 save_folder, # Ordnerpfad zum Speicherort, es werden keine Ergebnisse überschrieben, sondern dort neue Ornder erzeugt 
                 early_stoppping_model : bool,
                 patience_model, # Anzahl der Iterationen ohne Verbesserung bevor early stopping triggert. 
                 epochs_model : float, # Anzahl der Trainingsiteration innerhalb eines Boostings, durch Earlystopping meist deutlich weniger Iterationen
                 learningrate_model : float, # Lernrate Modell
                 batch_size_model : int, 
                 early_stoppping_TraDaBoostR2 : bool,
                 learningrate_TraDaBoostR2 : float, # Lernrate Tradaboost
                 n_estimators_tradaBoostR2 : float): # ANzahl an Boostingiterationen
        """
        Diese Klasse initialisiert und trainiert ein Modell mit der Tradaboost Methode.
        Es werden die zentralen Parameter des Trainings am Anfang gesetzt. 
        Danach können für dieses Modell Trainings durchgeführt werden, bei denen die Anteile der source und traget Domain Daten variiert werden. 
        Durch Setter diese definieren und dann trainieren."""

        # Ordner und Dateien
        self.model_path=model_path
        self.save_folder=save_folder       

        # Trainignsparameter des Modells
        self.early_stoppping_model = early_stoppping_model
        self.patience_model = patience_model
        self.epochs_model = epochs_model
        self.learningrate_model = learningrate_model
        self.batch_size_model = batch_size_model
        
        # TrAdaBoostR2 Trainingparameter 
        self.early_stoppping_TraDaBoostR2 = early_stoppping_TraDaBoostR2
        self.learningrate_TraDaBoostR2 = learningrate_TraDaBoostR2
        self.n_estimators_tradaBoostR2 = n_estimators_tradaBoostR2

        # Prozesszustände
        self.data_is_setted=False

        # Callbacks des Modells (innen, das ML-Modell NICHT Tradaboost)
        self.used_callbacks_model = []

        # Dokumentation der Trainingsparameter erstellen
        self.add_trianing_dokumentation()

    def add_trianing_dokumentation(self):
        """Fügt der Doku die Parameter mit denen das Training begonnen wird hinzu."""
        doku_training_initial = {
            "Training Parameter": {
                "Optimizer": {
                    "name": "Adam",
                    "learning_rate": self.learningrate_model
                },
                "early_stopping": {
                    "enabled": self.early_stoppping_model,
                    "patience": self.patience_model,
                    "restore_best_weights": True
                },
                "KerasRegressor": {
                    "epochs": self.epochs_model,
                    "batch_size": self.batch_size_model
                },
                "loss": "mean_absolute_error",
                "metrics": ["mae"]
            },
            "Parameter TradaBoostR2": {
                "early_stopping": self.early_stoppping_TraDaBoostR2,
                "n_estimators": self.n_estimators_tradaBoostR2,
                "learning_rate": self.learningrate_TraDaBoostR2
            }
        }
        self.doku_training_initial = doku_training_initial

    def get_training_dokumentation(self):
        return self.doku_training_initial

    def set_data_for_training(self, x_source, x_target, x_val, y_val, y_source, y_target, output_target: OutputTarget = None):
        # Datensätze definieren
        self.x_source = x_source
        self.x_target = x_target
        self.x_val = x_val
        self.y_val = y_val
        self.y_source = np.squeeze(y_source)
        self.y_target = np.squeeze(y_target)
        if not output_target is None:
            output_idx = output_target.get_index()
            self.y_source = self.y_source[:,output_idx]
            self.y_target = self.y_target[:,output_idx]
                    
        self.data_is_setted=True # Daten wurden gesetzt, Kontrolle hier nicht vorhanden, einfach aufpassen

    def _build_tradaBoostR2_model(self, output_target: Optional[OutputTarget] = None, this_learning_rate=None):
        """
        ACHTUNG: Die Targetdaten werden hier bereits definiert
        Gibt ein TradaBoostR2 Modell zurück, das für das Training verwendet werden kann.
        Verwendet ausschließlich die bei der Initialisierung gesetzten Objektparameter.
        """
        # EarlyStopping für Modell
        if self.early_stoppping_model:
            early_stopping = EarlyStopping(monitor="val_loss", patience=self.patience_model, restore_best_weights=True)
            self.used_callbacks_model.append(early_stopping)
        
        #Unterschiedliche Outputstukturen für da szusammengefasste Modell und die einzelnen Modelle
        # Modellfunktion abhängig vom Output-Target
        if (output_target is not None and 
            this_learning_rate is not None): # Leraningrate muss aus dem json Fild des Tuning genomen werden!!! # TODO Rdundant zu dem Training der single_models
            
            model_fn = lambda: build_model_single_output(
                model_path=os.path.join(self.model_path,output_target.get_output_name(),"best-model.h5"),
                output_type=output_target,
                learning_rate=this_learning_rate
            )
        else:
            model_fn = lambda: build_model_combinedoutputs(
                self.model_path,
                self.learningrate_model
            )

        # Keras Wrapper, weil adapt keras benutzt
        regressor = KerasRegressor(
            #model= build_model_combinedoutputs(self.model_path, self.learningrate_model),
            model=model_fn,
            epochs=self.epochs_model,
            batch_size=self.batch_size_model,
            callbacks=self.used_callbacks_model,
            verbose=1
        )

        self.tradaboost_model = TrAdaBoostR2(
            estimator=regressor,
            n_estimators=self.n_estimators_tradaBoostR2,
            lr=self.learningrate_TraDaBoostR2,
            Xt=self.x_target,
            yt=self.y_target,
            verbose=1
        )
        
    def execute_with_processdoku(self, output_target: Optional[OutputTarget] = None, this_learning_rate= None):
        """Trainiert ein Modell mit TrAdaBoostR2.
        Processdoku: In jeder iteration von TradaboostR2 wird der Trainingsfortschritt dokumentiert.
        Problem: Das dauert länger, und ich war nach sichten der Ergebnisse nicht unbedingt schlauer. 
        """

        start = time.time()

        # Tradaboost modell erzeugen
        self._build_tradaBoostR2_model(output_target=output_target, this_learning_rate=this_learning_rate)

        # Training mit Fortschrittsanzeige
        training_progress = []
        with tqdm(total=self.n_estimators_tradaBoostR2, desc="TrAdaBoostR2 Training") as pbar:
            for i in range(self.n_estimators_tradaBoostR2):
                # Callback für Dokumentation des Trainingprocesses
                csv_logger = CSVLogger(os.path.join(self.save_folder, f"training_log_iter{i}.csv"), append=False)
                self.used_callbacks_model.append(csv_logger)
                
                self.tradaboost_model.fit(
                    self.x_source,
                    self.y_source,
                    validation_data=(self.x_val, self.y_val),
                    callbacks=[*self.used_callbacks_model, csv_logger])
                model_i = self.tradaboost_model.estimators_[-1].model_
                loss = model_i.evaluate(self.x_val, self.y_val, verbose=0)
                progress = {
                    "iteration": i + 1,
                    "loss": loss,
                    "estimator_count": len(self.tradaboost_model.estimators_),
                    "errors": self.tradaboost_model.estimator_errors_,
                }
                training_progress.append(progress)

                # Modell abspeichern
                model_i.save(os.path.join(self.save_folder, f"model_of_boosting_iter_{i}.h5"))
                pbar.update(1)

        # Modell speichern
        self.final_model = self.tradaboost_model.estimators_[-1].model_
        self.final_model.save(os.path.join(self.save_folder, f"final_model.h5"))

        # Loggen
        with open(os.path.join(self.save_folder, "training_progress.json"), "w") as f:
            json.dump(training_progress, f, indent=4)

        end = time.time()
        print(f"Training abgeschlossen in {end - start:.2f}s")

    def execute_without_process_doku(self, output_target: Optional[OutputTarget] = None, this_learning_rate=None):
        """
        Die Variablen diehnen zur Definition von dem Tradaboosttraining mit dem einzelnen modellen, 
        TODO Ggf. eine andere Lösung finden
        """
        start = time.time()

        # Modell erstellen
        self._build_tradaBoostR2_model(output_target=output_target, this_learning_rate=this_learning_rate)

        # Modell trainieren
        self.tradaboost_model.fit(self.x_source, self.y_source,validation_data=(self.x_val,self.y_val))
        
        # Speichere finale Komponenten als Objektattribute
        self.tradaboost_model = self.tradaboost_model
        self.final_model = self.tradaboost_model.estimators_[-1].model_
        self.training_progress = []  # Leere Doku (für Kompatibilität mit save_model())

        end= time.time()
        print(f"Training abgeschlossen in {end - start:.2f}s")
        
    @staticmethod
    def save_model(final_model,save_folder):
        """
        Diese Methdode speichert das Modell. 
        Die Dokumentation des Trainings, falls gewünscht, wird beim trianing abgespeichert.
        ACHTUNG: Ergebnisse könne überschrieben werden, wenn die Dateinamen gleich sind.
        """
        
        # Modell speichern
        final_model.save(os.path.join(save_folder, "best_model.h5"))
