import json
import os

from src.model import tradaBoostR2_setup
from src.data import NPY


class TrAdaBoostR2TrainingExecution:
    """Diese Klasse baut ein Training mit TrAdaBoostR2 auf.
    """
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
        """
        ACHTUNG: Das modell mit den Parametern, die das Training definieren sind in dem Objekt tradaboostr2_model enthalten.
        ACHTUNG: die x-Daten (input des Modells) sind IMMER skaliert. 
        Hier werdem diesen Modell die Daten zugewiesen und dann trainiert.
        Nach dem Training wird die doku und das modell abgespeichert.
        """
        
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

        #Dokumentation des Training smit Tradaboost
        initial_training_setup = {
                "Speicherorte": {
                "Modell": model_file,
                "Simulationsdaten": dateipfad_source_daten,
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
        
        # Abspeichern des Initial Setups
        json_path = os.path.join(save_folder, f"tradaBoostR2_training_BasisParameter.json")
        with open(json_path, 'w') as json_file:
            json.dump(initial_training_setup, json_file, indent=4, default=str)

        #Daten dem Trainingobjekt übergeben
        tradaboostr2_model.set_data_for_training(
            x_source=x_source,
            y_source=y_source,
            x_target=x_target,
            y_target=y_target,
            x_val=x_source_test,
            y_val=y_source_test)
        
        # Training starten
        if training_dokumentation:
            tradaboostr2_model.execute_with_processdoku()
        else:
            tradaboostr2_model.execute_without_process_doku()

        # Modell speichern
        print(tradaboostr2_model)
        tradaBoostR2_setup.TrAdaBoostR2ModelSetup.save_model(
            final_model=tradaboostr2_model.final_model,
            #training_progress=tradaboostr2_model.get_training_dokumentation(),
            save_folder=tradaboostr2_model.save_folder,
        )      