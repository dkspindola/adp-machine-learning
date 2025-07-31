import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib

import src.execution
import src.model
import src.util

importlib.reload(src.execution)
importlib.reload(src.model)
importlib.reload(src.util)

from src.experiment.multiple_trAdaBoostR2_trianings_specific_models import MultipleTrAdaBoostR2TrainingSpecificModleOutput

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    os.chdir(project_root)  # Arbeitsverzeichnis setzen
    
    dateipfad_femDaten="build\\window_split\\sim_data_preprocessed\\1744139600\\"
    dateipfad_realDaten ="build\\window_split\\real-data\\Realdaten_5Prozent_Random_N15\\"
    model_file = os.path.join("assets", "models", "untrained", "seperated-models")
    save_folder = os.path.join("build", "tradaboost_model_specifiv_for_output", "trAdaBoostR2_1_Realdaten_5Prozent_N15")
    print("Starte Training mit:")
    print(f"Source: {dateipfad_femDaten}")
    print(f"Target: {dateipfad_realDaten}")
    MultipleTrAdaBoostR2TrainingSpecificModleOutput(target_data_folder=dateipfad_realDaten, 
                                                    source_data_folder=dateipfad_femDaten,
                                                    folder_base_model=model_file,
                                                    result_folder=save_folder,
                                                    result_folder_name_initial="trAdaBoostR2_1_SpecificModel")

if __name__ == "__main__":
    main()
