from src.model.cnn import CNN
import json
import os   
from src.util import timestamp

class CNNValidationExecution:
    """Handles the validation execution of a CNN model.

    This class provides a method to validate a CNN model using a specified
    dataset and saves the validation results to a JSON file.
    """

    @classmethod
    def execute(cls, model_folder: str, data_folder: str):
        """Executes the validation of a CNN model.

        This method locates the CNN model file in the specified folder,
        validates it using the provided dataset, and saves the results
        to a JSON file in a structured output folder.

        Args:
            model_folder (str): The folder containing the CNN model file (.h5).
            data_folder (str): The folder containing the dataset for validation.

        Raises:
            FileNotFoundError: If no .h5 file is found in the model folder.
        """
        model_file = None
        for file in os.listdir(model_folder):
            if file.endswith('.h5'):
                model_file = os.path.join(model_folder, file)
                break
        
        if model_file is None:
            raise FileNotFoundError("No .h5 file found in the model folder")

        cnn = CNN.from_file(model_file)
        content = {}
        results = cnn.validate(data_folder)
        content['data'] = data_folder   
        content['model'] = model_file   
        content['results'] = results    
        folder = os.path.join('build', 'validate', os.path.split(os.path.split(model_folder)[0])[1], timestamp())
        os.makedirs(folder, exist_ok=True)
        json.dump(content, open(os.path.join(folder, 'validation_results.json'), 'w'), indent=4)

