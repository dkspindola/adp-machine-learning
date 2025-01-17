import os
from src.process import WindowSplitting
from src.util import timestamp

class WindowSplittingExecution:
    @classmethod
    def execute(cls, data_file: str, 
                     batch_split: bool=False, 
                     validation_split: bool=True, 
                     test_size: float=0.2, 
                     seed: int=42, 
                     batchsize: int=1800,
                     interpolation: bool=False,
                     window_size: int=10
                     ):
        
        process = WindowSplitting(batch_split, validation_split, interpolation)

        process.load(data_file)
        process.start(test_size, seed, batchsize, window_size)
        process.save(os.path.join('build', 'split', timestamp()))