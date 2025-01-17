from src.execution import CNNTrainingExecution
from src.execution import WindowSplittingExecution
from numpy import random
import os

class MultipleCNNTrainingExperiment:
    @classmethod
    def start(cls, N: int, test_size: float, model_file: str, data_folder: str):
        seed: list[int] = random.randint(0, 100, N)

        for n in range(N):
            WindowSplittingExecution.execute(data_folder, 
                                             batch_split= False, 
                                             validation_split=True, 
                                             test_size=test_size, 
                                             seed=seed[n], 
                                             batchsize=1800, 
                                             interpolation=False, 
                                             window_size=10)
        
        #data_file: list[str] = os.listdir('build/split')

        #CNNTrainingExecution.execute(model_file, data_folder)