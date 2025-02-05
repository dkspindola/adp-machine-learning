import os
import json
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

        _, data_name = os.path.split(data_file)
        data_name, _ = os.path.splitext(data_name)

        metadata = {
            "data": data_file, 
            "seed": int(seed),
            "test_size": test_size,
            "window_size": window_size,
            "validation_split": validation_split,
            "batch_split": batch_split,
            "interpolation": interpolation,
            "batch_size": batchsize
        }

        folder = os.path.join('build', 'window_split', data_name, timestamp())

        process.save(folder)
        json.dump(metadata, open(os.path.join(folder, 'metadata' + '.json'), 'w'), indent=4)
