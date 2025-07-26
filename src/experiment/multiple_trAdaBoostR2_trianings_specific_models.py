import os
from src.execution import TrAdaBoostR2ModelSetup_SingleModel_Execution
from src.model import tradaBoostR2_setup

class MultipleTrAdaBoostR2TrainingSpecificModleOutput:
    """
    TrAdaBoostR2 Training auf den Modellen mit je einem Modell für ein Output.
    """
    def __init__(self, target_data_folder, source_data_folder, folder_base_model,result_folder, result_folder_name_initial):
        self.target_data_folder = target_data_folder
        self.source_data_folder = source_data_folder
        self.folder_base_model=folder_base_model
        self.result_folder = result_folder
        self.result_folder_name_initial = result_folder_name_initial

        self._execute_all_trainings()

    def _execute_all_trainings(self):
        target_data_subfolders = [
            os.path.join(self.target_data_folder, name) for name in os.listdir(self.target_data_folder)
            if os.path.isdir(os.path.join(self.target_data_folder, name))
        ]
        count = 0
        total = len(target_data_subfolders)
        padding_width = len(str(total))

        # Liste mit den Source Daten Folders


        for this_target_data in target_data_subfolders:
            count += 1
            run_name = f"{self.result_folder_name_initial}_{str(count).zfill(padding_width)}"
            save_path = os.path.join(self.result_folder, run_name)
            print(save_path)
            os.makedirs(save_path, exist_ok=True)
            print(f"Target Daten: {this_target_data}")
        
                        
            modell=tradaBoostR2_setup.TrAdaBoostR2ModelSetup(
                model_path=self.folder_base_model,
                early_stoppping_model=True,
                patience_model=3,
                learningrate_model=None, # Daten werden bei None aus der Doku genommen
                n_estimators_tradaBoostR2=5,
                save_folder=save_path,
                epochs_model=10,  # Wert bitte setzen
                batch_size_model=32,  # Wert bitte setzen
                early_stoppping_TraDaBoostR2=False,  # Wert bitte setzen
                learningrate_TraDaBoostR2=1  # Wert bitte setzen
            )

            training=TrAdaBoostR2ModelSetup_SingleModel_Execution(tradaboostr2_model=modell,
                                                            dateipfad_source_daten=self.source_data_folder,
                                                            dateipfad_target_daten=f"{this_target_data}\\",
                                                            use_scaled_labels=True,
                                                            model_file=self.folder_base_model,
                                                            save_folder=save_path,
                                                            save_filename="testmodel",
                                                            training_dokumentation=False)