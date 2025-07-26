import os
import json
from tensorflow import keras
from tensorflow.keras.callbacks import Callback,EarlyStopping, CSVLogger 
from src.model.build_model_for_TrAdaBoostR2 import build_model_single_output 
from src.data import NPY, OutputTarget
from src.util import timestamp
from src.model import tradaBoostR2_setup
from src.data import NPY

class TrAdaBoostR2ModelSetup_SingleModel_Execution:
    """Ausführen des TradaboostR2 Trainings der Modelle mit je einem Modell für ein Output."""   
    def __init__(self,
                tradaboostr2_model : tradaBoostR2_setup.TrAdaBoostR2ModelSetup,
                dateipfad_source_daten, # Hier die FEM Daten 
                dateipfad_target_daten, # Hier die Realdaten 
                training_dokumentation : bool, # Wenn true, dann wird in jeder Iteration ein Outpunt geschireben, in dem der Trainginprozess dokumentiert ist (Zeitintensiv)
                use_scaled_labels, # Definition ob skalierte oder unskalierte Labels verwendet werden sollen, 
                #Hinweis: Sakliert sollte besser funktionieren (Boosting mit Interpolation oder Extrapolation))
                
                model_file, # Dateipfad zum Modell, h5 Datei
                save_folder, # Speicherordner des Mdoells 
                save_filename, # Speichername des Modells
                ): 
        
        # TODO Redundant zu TrAdaBoostR2TrainingExecution verringern, Basisklasse implementieren

        # Auswahl, ob akalierte oder unskalierte Label
        if use_scaled_labels:
            label_selcetion="-scaled"
        else:
            label_selcetion=""
        
        # Source Daten 
        x_source = NPY.from_file(dateipfad_source_daten + "x-train-scaled.npy").array
        y_source = NPY.from_file(dateipfad_source_daten +  f"y-train{label_selcetion}.npy").array
        x_source_test = NPY.from_file(dateipfad_source_daten + "x-validate-scaled.npy").array
        y_source_test = NPY.from_file(dateipfad_source_daten +f"y-validate{label_selcetion}.npy").array
        
        # Target Daten
        x_target = NPY.from_file(dateipfad_target_daten + "x-train-scaled.npy").array
        y_target = NPY.from_file(dateipfad_target_daten + f"y-train{label_selcetion}.npy").array
        timestamp_str = timestamp() # Zeitstempel vor schleife, sollen alle in einem Ordner landen
        for output_target in OutputTarget:
            #Save Folder erzeugen, weil Name des Outputs beeinhaltet sein MUSS!!!
            if save_folder:
                model_save_folder = os.path.join(save_folder,output_target.get_output_name())
            else:
                model_save_folder = os.path.join('build', 'tradaboost_model_specifiv_for_output', timestamp_str, output_target.get_output_name())
            os.makedirs(model_save_folder, exist_ok=True)

            # Einlesen des Model File, um das Modell zu bekommen und die Learningrate
            this_model_file= os.path.join(model_file, output_target.get_output_name(), 'best-model.h5')
            # EInlesen des Doku zum Tuning, um Learningrate zu bekommen.
            hyperparam_path = os.path.join(model_file, output_target.get_output_name(), "best-hyperparameters.json")
            with open(hyperparam_path, 'r') as f:
                hyperparams = json.load(f)
            learning_rate = hyperparams["learning_rate"]
            if learning_rate is None:
                raise ValueError(f"Die Leraningrate darf nicht None sine!!! \n siehe {hyperparam_path}")
            #ACHTUNG, total BS, hier die learning_rate des einzelnen Modells setzen!!!
            #Dokumentation des Training smit Tradaboost
            initial_training_setup = {
                    "Speicherorte": {
                    "Modell": this_model_file,
                    "Simulationsdaten": dateipfad_source_daten,
                    "this_learning_rate": learning_rate, 
                    "Single Model": output_target.get_output_name(),
                    "Realdaten": dateipfad_target_daten,
                    "save_folder": save_folder,
                    "save_filename": save_filename,
                    "Daten":{
                        "source_date": dateipfad_source_daten,
                        "target_data": dateipfad_target_daten,
                    }
                    },
                    "Dimensionen der Daten": { # Dimensionen abspeichern um Datenverhältnis zu kontrollieren
                        "x_source": x_source.shape,
                        "y_source": y_source.shape,
                        "x_target": x_target.shape,
                        "y_target": y_target.shape,
                        "X_source_test": x_source_test.shape,
                        "y_source_test": y_source_test.shape
                    },
                    "Training Parameter": tradaboostr2_model.get_training_dokumentation() # Hinzufügen der Dokumentation der Trainingsparameter                   
                }
            
            # Abspeichern des Initial Setups im selben Pfad wie das model, macht das Sinn? Dort viel doppelt, aber Learningrate ist einzeln
            json_path = os.path.join(model_save_folder, f"tradaBoostR2_training_BasisParameter.json")
            print(f"Speichern der Doku in: {json_path}")
            with open(json_path, 'w') as json_file:
                json.dump(initial_training_setup, json_file, indent=4, default=str)

            #Daten dem Trainingobjekt übergeben
            tradaboostr2_model.set_data_for_training(
                x_source=x_source,
                y_source=y_source,
                x_target=x_target,
                y_target=y_target,
                x_val=x_source_test,
                y_val=y_source_test, 
                output_target=output_target)
            
            # Training starten
            # TODO: Total BS, hier die Laerning Rate übergeben, da diese zu den Modell mit dem Output aus der doku zugeordnet weren muss
            if training_dokumentation:
                tradaboostr2_model.execute_with_processdoku(output_target=output_target,this_learning_rate=learning_rate)
            else:
                tradaboostr2_model.execute_without_process_doku(output_target=output_target, this_learning_rate=learning_rate)

            # Modell speichern
            print(tradaboostr2_model)
            tradaBoostR2_setup.TrAdaBoostR2ModelSetup.save_model(
                final_model=tradaboostr2_model.final_model,
                #training_progress=tradaboostr2_model.get_training_dokumentation(),
                save_folder=model_save_folder, # TODO hier den Ordner für 
            )
