
from src.execution import CNNTrainingExecution
from numpy import random
import os

class MultipleCNNTrainingOnExistingData:
    """
    """
    def __init__(self, target_folder, folder_data, folder_base_model, result_folder_name_initial):
        self.target_folder = target_folder
        self.folder_data = folder_data
        self.folder_base_model = folder_base_model
        self.result_folder_name_initial = result_folder_name_initial

        self._execute_all_trainings()

    def _execute_all_trainings(self):
        data_subfolders = [
            os.path.join(self.folder_data, name) for name in os.listdir(self.folder_data)
            if os.path.isdir(os.path.join(self.folder_data, name))
        ]

        count = 0
        total = len(data_subfolders)
        padding_width = len(str(total))

        for data_path in data_subfolders:
            count += 1
            run_name = f"{self.result_folder_name_initial}_{str(count).zfill(padding_width)}"
            save_path = os.path.join(self.target_folder, run_name)
            print(save_path)
            os.makedirs(save_path, exist_ok=True)

            CNNTrainingExecution.execute_three_models_training(model_folder=self.folder_base_model,
                                                               data_file=data_path,
                                                               save_folder=save_path)
   