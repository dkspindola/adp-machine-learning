import os
import json
from src.process import WindowSplitting
from src.util import timestamp

class WindowSplittingExecution:
    """Handles the execution of the window splitting process.

    This class provides a method to execute the window splitting process
    on a given dataset, save the processed data, and generate metadata.
    """

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
                     decimal: str):
        """Executes the window splitting process.

        Args:
            data_file (str): Path to the input data file.
            batch_split (bool): Whether to split data into batches.
            validation_split (bool): Whether to create a validation split.
            test_size (float): Proportion of the dataset to include in the test split.
            seed (int): Random seed for reproducibility.
            batchsize (int): Size of each batch if batch splitting is enabled.
            interpolation (bool): Whether to apply interpolation to the data.
            window_size (int): Size of the window for splitting.
            sep (str): Delimiter used in the input data file.
            decimal (str): Decimal separator used in the input data file.

        Returns:
            None
        """
        
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
