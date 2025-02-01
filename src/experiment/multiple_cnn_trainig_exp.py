from src.execution import CNNTrainingExecution
from src.execution import WindowSplittingExecution
from numpy import random
import os

class MultipleCNNTrainingExperiment:
    @classmethod
    def start(cls, N: int, test_size: float, model_file: str, data_file: str, learning_rate: float, generate_new_split: bool):
        if generate_new_split:
            seed: list[int] = random.randint(0, 100, N)
            
            for n in range(N):
                WindowSplittingExecution.execute(data_file, 
                                                 batch_split= False, 
                                                 validation_split=True, 
                                                 test_size=test_size, 
                                                 seed=seed[n], 
                                                 batchsize=1800, 
                                                 interpolation=False, 
                                                 window_size=10)
        
        data_file: list[str] = os.listdir('build/split')
        data_file.sort(key=int, reverse=True)

        for n in range(N):
            CNNTrainingExecution.execute(model_file, os.path.join('build/split', data_file[n]), 'trained_model', learning_rate)