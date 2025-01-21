from src.model.cnn import CNN
import json
import os   
from src.util import timestamp

class CNNValidationExecution:
    @classmethod
    def execute(cls, model_folder: str, data_folder: str):
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
        folder = os.path.join('build', 'validate', timestamp())
        os.makedirs(folder, exist_ok=True)
        json.dump(content, open(os.path.join(folder, 'validation_results.json'), 'w'), indent=4)

