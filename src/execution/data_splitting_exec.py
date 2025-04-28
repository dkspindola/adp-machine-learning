import os
import json
from src.util import timestamp
from src.process import DataSplitting


class DataSplittingExecution:
    """Handles the execution of the data splitting process.

    This class provides a method to execute data splitting, including
    options for batch splitting, validation splitting, and saving metadata.
    """

    @classmethod
    def execute(cls, data_file: str, 
                     batch_split: bool = False, 
                     validation_split: bool = True, 
                     test_size: float = 0.2, 
                     seed: int = 42, 
                     batchsize: int = 1800):
        """Executes the data splitting process.

        Args:
            data_file (str): Path to the input data file.
            batch_split (bool): Whether to split the data into batches. Defaults to False.
            validation_split (bool): Whether to create a validation split. Defaults to True.
            test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.2.
            seed (int): Random seed for reproducibility. Defaults to 42.
            batchsize (int): Size of each batch if batch splitting is enabled. Defaults to 1800.
        """
        
        process = DataSplitting(batch_split, validation_split)

        process.load(data_file)
        process.start(test_size, seed, batchsize)
        
        _, data_name = os.path.split(data_file)
        data_name, _ = os.path.splitext(data_name)

        metadata = {
            "data": data_file, 
            "seed": seed,
            "test_size": test_size,
            "validation_split": validation_split,
            "batch_split": batch_split,
            "batch_size": batchsize
        }

        folder = os.path.join('build', 'split', data_name, timestamp())

        process.save(folder)
        json.dump(metadata, open(os.path.join(folder, 'metadata' + '.json'), 'w'), indent=4)
