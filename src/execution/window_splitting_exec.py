import os
import json
from src.process import WindowSplitting
from src.util import timestamp

class WindowSplittingExecution:
    @classmethod
    def execute(cls, data_file: str, 
                     batch_split: bool, 
                     validation_split: bool, 
                     test_size: float, 
                     seed: int, 
                     batchsize: int,
                     interpolation: bool,
                     window_size: int,
                     sep: str,
                     decimal: str                  
                     ):
        
        process = WindowSplitting(batch_split, validation_split, interpolation)
        process.load(data_file, sep, decimal)
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
        print(f"Data saved in {folder}")
