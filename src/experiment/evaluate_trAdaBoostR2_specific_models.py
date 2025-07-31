
import json
import os
from src.model import CNN
from tensorflow import keras
import pickle
from src.data import NPY
import numpy as np
from src.data.output_type import OutputTarget

class EvaluateTrAdaBoostR2:
    """Klasse, um mit tradaboost trainierte  Modelle zu evaluieren.
    Besonderheit: Rückscalierung muss durchgeführt werden, um die Fehler MAE mit Einheiten zu bestimmen."""    
    def __init__(self):
        pass

    @staticmethod
    def process_all_results(base_dir: str, use_scaled_labels: bool = True):
        """
        Durchläuft alle Unterordner in `base_dir` und alle OutputTarget-Ordner darin,
        lädt Modell + JSON, lädt Testdaten, evaluiert und schreibt pro Output eine result.json.
        """
        all_results = {}
        mae_list=[]
        real_data_folder=""
        # 1. Jede Real‑Daten‑Konfiguration
        for real_cfg in os.listdir(base_dir):
            real_cfg_path = os.path.join(base_dir, real_cfg)
            if not os.path.isdir(real_cfg_path):
                continue

            cfg_results = {}
            preds_cols=[] # Jedes Listenelement ein Skaliertes Ergebnis.
            # 2. Jeder OutputTarget-Unterordner
            for output in OutputTarget:
                out_name = output.get_output_name()
                out_path = os.path.join(real_cfg_path, out_name)
                model_file = os.path.join(out_path, "best_model.h5")
                json_file  = os.path.join(out_path, "tradaBoostR2_training_BasisParameter.json")

                if not (os.path.exists(model_file) and os.path.exists(json_file)):
                    print(f"> Fehlende Dateien in {out_path}, überspringe.")
                    continue

                # 2.1 JSON einlesen
                with open(json_file, "r", encoding="utf-8") as f:
                    params = json.load(f)

                # 2.2 Test-Daten laden
                real_data_folder = params["Speicherorte"]["Realdaten"]
                # je nach Skalierung
                x_test = NPY.from_file(os.path.join(real_data_folder, "x-test-scaled.npy")).array
                y_test = NPY.from_file(os.path.join(
                    real_data_folder,
                    "y-test-scaled.npy" if use_scaled_labels else "y-test.npy"
                )).array

                # Nur die Spalte des aktuellen OutputTarget wählen, als (N,1)
                idx = output.get_index()
                y_test = y_test[:, idx:idx+1]
   
                # 2.4 Modell laden
                #model = load_model(model_file, compile=False)
                model = CNN.from_file(model_file)

                # 2.5 Vorhersage und Rücktransformation
                preds_cols.append(model.model.predict(x_test))  # erwartet (N,1)
                # Rücktransformation nur für diese Spalte
            print(os.path.join(real_data_folder, "scalers_labels.pkl"))
            with open(os.path.join(real_data_folder, "scalers_labels.pkl"), "rb") as f:
                        scalers_labels = pickle.load(f)

            preds_full = np.hstack(preds_cols)  # Spalten: [X, Y, Phi]
            errs_scaled = EvaluateTrAdaBoostR2._compute_errors(preds_full, y_test)

            # 5b) Nur rückskalieren, wenn use_scaled_labels=True
            if use_scaled_labels:
                xy_orig  = scalers_labels[0].inverse_transform(preds_full[:, :2])
                phi_orig = scalers_labels[1].inverse_transform(preds_full[:, 2:3])
                preds_orig = np.hstack([xy_orig, phi_orig])

                #yxy_orig  = scalers_labels[0].inverse_transform(y_test[:, :2])
                #yphi_orig = scalers_labels[1].inverse_transform(y_test[:, 2:3])
                #y_orig    = np.hstack([yxy_orig, yphi_orig])
                y_orig=NPY.from_file(os.path.join(
                    real_data_folder,
                    "y-test-scaled.npy" if use_scaled_labels else "y-test.npy"
                )).array
                print("preds_full.shape:", preds_full.shape)
                print("y_test.shape:", y_test.shape)
                errs_orig = EvaluateTrAdaBoostR2._compute_errors(preds_orig, y_orig)
            else:
                errs_orig = None

            # 6) Speichere Ergebnisse
            result = {
                "scaled_errors": errs_scaled,
                "orig_errors": errs_orig
            }

            # Schreibe eine JSON in die Modelldatei
            out_json = os.path.join(real_cfg_path, "evaluation_results.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)

            cfg_results[out_name] = result

        # Aggregiere alle drei Outputs unter der Real-Konfiguration
        all_results[real_cfg] = cfg_results


        return all_results
    @staticmethod
    def _compute_errors(preds: np.ndarray, y: np.ndarray):
        """MAE/MSE/RMSE je Spalte. preds,y in gleicher Form (N,3)."""
        errors = {}
        print("preds_full.shape:", preds.shape)
        print("y_test.shape:", y.shape)
        
        for i, out in enumerate(OutputTarget):    
            diff = preds[:, i] - y[:, i]
            mae  = float(np.mean(np.abs(diff)))
            mse  = float(np.mean(diff**2))
            rmse = float(np.sqrt(mse))
            errors[out.value] = {"MAE": mae, "MSE": mse, "RMSE": rmse}
        return errors
    """
    @staticmethod
    @register_keras_serializable(package="custom_layers")# Hier zuvor den Import bei der Klasse ausführen (sihe oben)
    def process_all_results(base_dir,trained_on_scaled_Labels=None,tradaboost_model=None):
        
        Durchläuft alle Unterordner im angegebenen Basisverzeichnis und führt die Evaluierung durch.
        Für jeden Unterordner wird die JSON-Datei und das Modell geladen, und die Evaluierung wird durchgeführt.
        Die Ergebnisse werden in einer Liste gespeichert und zurückgegeben.

        results = []  # Liste zur Speicherung aller Ergebnisse
        # Durchlaufe alle Einträge im Basisverzeichnis
        for entry in os.listdir(base_dir):
            subfolder = os.path.join(base_dir, entry)
            print("==================================================")
            print(f"Subfolder: {subfolder}")
            if os.path.isdir(subfolder):
                # Pfade zu den benötigten Dateien im aktuellen Unterordner
                json_file = os.path.join(subfolder, "tradaBoostR2_training_BasisParameter.json")
                model_file = os.path.join(subfolder, "best_model.h5")
                
                # Prüfe, ob die erforderlichen Dateien existieren
                if os.path.exists(json_file) and os.path.exists(model_file):
                    # Einlesen der JSON-Datei
                    with open(json_file, "r", encoding="utf-8") as f:
                        params = json.load(f)
                    # Einlesen des Modells
                    print(f"Einlesen des Modells: {model_file}")
                    model = CNN.from_file(model_file)
                    
                    #Fallunterscheidung ob auf skalierten Labels trainiert wurde oder nicht
                    #Es wurde keine Angabe beim AUfruf gemacht, also über Dokumentation überprüfen
                    if trained_on_scaled_Labels==None:
                        #Prüfe, ob auf scaled Labels trainiert wurde
                        labels_pfad=params.get("Daten",{}).get("y_source_scaled", None)
                        trained_on_scaled_Labels = labels_pfad.endswith("-scaled.npy")
                    
                    #Scaler einlesen, immer machen, weil sinnvoll für Kontrolle
                    if tradaboost_model==None or tradaboost_model==True:
                        real_data_path = params.get("Speicherorte", {}).get("Realdaten", None)
                    elif tradaboost_model==False:
                        real_data_path = params.get("Speicherorte", {}).get("Daten", None)
                    print(f"Einlesen der Scaler: {real_data_path}")
                    with open(os.path.join(real_data_path, "scalers_labels.pkl"), "rb") as f:
                        scalers_labels = pickle.load(f)

                    #Fallunterscheidung um Daten für Evaluation zu laden
                    if trained_on_scaled_Labels:
                        x_test_file = os.path.join(real_data_path, "x-test-scaled.npy")
                        y_test_file = os.path.join(real_data_path, "y-test-scaled.npy")
                    else:
                        x_test_file = os.path.join(real_data_path, "x-test-scaled.npy")
                        y_test_file = os.path.join(real_data_path, "y-test.npy")

                    #Evaluieren des Modells
                    if os.path.exists(x_test_file) and os.path.exists(y_test_file):
                        # Annahme: NPY.from_file gibt ein Objekt zurück, dessen .array das eigentliche NumPy-Array ist.
                        X_target_val_scaled = NPY.from_file(x_test_file).array
                        
                        y_target_val_scaled = NPY.from_file(y_test_file).array
                        # Evaluieren des Modells auf den Target-Validierungsdaten
                        evaluation = EvaluateTrAdaBoostR2.eval_model(model,trained_on_scaled_labels=trained_on_scaled_Labels,scalers_labels=scalers_labels, 
                                                X_val=X_target_val_scaled, y_val=np.squeeze(y_target_val_scaled))
                    else:
                        print(f"Testdaten nicht gefunden in: {real_data_path}")
                        evaluation = None

                    if tradaboost_model==True:
                        #real_data_path = params.get("Speicherorte", {}).get("Realdaten", None)
                        dimension_trainingdaten= params.get("Dimensionen der Daten", {}).get("X_source_scaled", None)
                        dimension_validationdaten= params.get("Dimensionen der Daten", {}).get("X_source_scaled_Test", None)
                        dimension_realdaten= params.get("Dimensionen der Daten", {}).get("X_target_scaled", None)
                    if tradaboost_model==False:
                        #real_data_path = params.get("Speicherorte", {}).get("Daten", None)
                        dimension_trainingdaten= params.get("Dimensionen der Daten", {}).get("X_scaled", None)
                        dimension_validationdaten= params.get("Dimensionen der Daten", {}).get("X_scaled_Test", None)
                        dimension_realdaten= ["Kein Tradaboost Modell, diese Feld hat hier keine Bedeutung"]

                    result_entry = {
                        "subfolder": entry,         # Name des Unterordners
                        #"parameters": params,       # Ausgelesene Parameter aus der JSON-Datei
                        "Anzahl Trainingsdaten": dimension_trainingdaten[0], # Anzahl der Trainingsdaten
                        "Anzahl Validierungsdaten": dimension_validationdaten[0], # Anzahl der Validierungsdaten
                        "Anzahl Realdaten": dimension_realdaten[0], # Anzahl der Realdaten
                        "evaluation": evaluation,   # Ergebnis der Evaluierung
                    }
                    #TODO hier abpeichern von result_entry.json in dem Ordner subfolder

                    result_entry_path = os.path.join(subfolder, "result_entry.json")
                    with open(result_entry_path, "w", encoding="utf-8") as f:
                        json.dump(result_entry, f, indent=4, ensure_ascii=False)

                    
                    results.append(result_entry)
                else:
                    print(f"Erforderliche Dateien fehlen in: {subfolder}")

        return results
    
    @staticmethod
    def eval_model(model,scalers_labels, X_val, y_val,trained_on_scaled_labels=False):
        TODO Harter COde, später verbessern
        model - Das trainierte Modell, das evaluiert werden soll.
        scaler - Scaler des Datensatzes
        X_val - Die Eingabedaten für die Validierung, sind skaliert
        y_val - Die Zielwerte für die Validierung, sind skaliert 
        

        # Vorhersagen in skalierter Form, z.B. shape=(n_samples, 3)
        predictions = model.model.predict(X_val)
        
        #Ergebnissdaten 
        results = {
            "scaled": {},
            "Not scaled": {}
        }

        def get_errors(predictions, y_val):
            # Berechnung der Fehler in der skalierten Skala
            error_X_scaled = np.abs(predictions[:, 0] - y_val[:, 0])
            error_Y_scaled = np.abs(predictions[:, 1] - y_val[:, 1])
            error_Phi_scaled = np.abs(predictions[:, 2] - y_val[:, 2])
            
            mae_X = np.mean(error_X_scaled)
            mae_Y = np.mean(error_Y_scaled)
            mae_Phi = np.mean(error_Phi_scaled)
            
            mse_X = np.mean((predictions[:, 0] - y_val[:, 0]) ** 2)
            mse_Y = np.mean((predictions[:, 1] - y_val[:, 1]) ** 2)
            mse_Phi = np.mean((predictions[:, 2] - y_val[:, 2]) ** 2)
            
            rmse_X = np.sqrt(mse_X)
            rmse_Y = np.sqrt(mse_Y)
            rmse_Phi = np.sqrt(mse_Phi)

            error={'Metrik': ['MAE', 'MSE', 'RMSE'],
                'Verstellweg_X': [mae_X, mse_X, rmse_X],
                'Verstellweg_Y': [mae_Y, mse_Y, rmse_Y],
                'Verstellweg_Phi': [mae_Phi, mse_Phi, rmse_Phi]
            }
            return error
        
        #Fallunterscheidung ob Skalierung der Labels vorliegt oder nicht
        if trained_on_scaled_labels:
            # Rücktransformation der skalieren Vorhersagen und der Zielwerte in die Originalskala.
            # Wir gehen davon aus, dass scaler zeilenweise angewendet wird:
            #Predictions zurückskalieren
            xy_scaled = predictions[:, :2]
            phi_scaled = predictions[:, 2:3]
            xy_original = scalers_labels[0].inverse_transform(xy_scaled)
            phi_original = scalers_labels[1].inverse_transform(phi_scaled)
            predictions_original = np.hstack([xy_original, phi_original])

            #Labels zurückskalieren
            y_xy_scaled = y_val[:, :2]
            y_phi_scaled = y_val[:, 2:3]
            y_xy_original = scalers_labels[0].inverse_transform(y_xy_scaled)
            y_phi_original = scalers_labels[1].inverse_transform(y_phi_scaled)
            y_val_original = np.hstack([y_xy_original, y_phi_original])

            #Fehler der zurückscalierten Daten, Fehler in Originalskala
            results["scaled"] = get_errors(predictions, y_val)
            results["Not scaled"] = get_errors(predictions_original, y_val_original)
        else:
            results["scaled"] ={}
            results["Not scaled"] = get_errors(predictions, y_val)

        return results
    """