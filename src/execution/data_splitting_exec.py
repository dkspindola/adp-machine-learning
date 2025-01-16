import os
from src.util import timestamp
from src.process import DataSplitting


class DataSplittingExecution:
    @classmethod
    def execute(cls, data_file: str, 
                     batch_split: bool=False, 
                     validation_split: bool=True, 
                     test_size: float=0.2, 
                     seed: int=42, 
                     batchsize: int=1800,
                     ):
        
        process = DataSplitting(batch_split, validation_split)

        process.load(data_file)
        process.start(test_size, seed, batchsize)
        process.save(os.path.join('build', 'split', timestamp()))