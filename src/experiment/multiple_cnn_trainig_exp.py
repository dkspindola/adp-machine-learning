from src.execution import CNNTrainingExecution
from src.execution import WindowSplittingExecution
from numpy import random
import os

class MultipleCNNTrainingExperiment:
    @classmethod
    def start(cls, N: int, test_size: float, model_file: str, data_file: str, learning_rate: float, generate_new_split: bool, sep: str, decimal: str, batchsize: int, batch_split: bool, interpolation: bool):
        if generate_new_split:
            seed: list[int] = random.randint(0, 32000, N)
            
            for n in range(N):
                WindowSplittingExecution.execute(data_file, 
                                                 batch_split=batch_split, 
                                                 validation_split=True, 
                                                 test_size=test_size, 
                                                 seed=seed[n], 
                                                 batchsize=batchsize, 
                                                 interpolation=interpolation, 
                                                 window_size=10,
                                                 sep=sep, 
                                                 decimal=decimal)
        
        _, data_name = os.path.split(data_file)
        data_name, _ = os.path.splitext(data_name)

        folder = os.path.join('build', 'window_split', data_name)

        data_files: list[str] = os.listdir(folder)
        data_files.sort(key=int, reverse=True)

        for n in range(N):
            CNNTrainingExecution.execute(model_file, os.path.join(folder, data_files[n]), 'cnn.h5', learning_rate)