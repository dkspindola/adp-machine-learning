from src.execution import CNNTrainingExecution
from src.execution import WindowSplittingExecution
from numpy import random
import os

class MultipleCNNTrainingExperiment:
    """Class to manage multiple CNN training experiments with data splitting."""

    @classmethod
    def start(cls, N: int, test_size: float, model_file: str, data_file: str, learning_rate: float, 
              generate_new_split: bool, sep: str, decimal: str, batchsize: int, batch_split: bool, 
              interpolation: bool):
        """Starts the multiple CNN training experiments.

        Args:
            N (int): Number of experiments to run.
            test_size (float): Proportion of the dataset to include in the test split.
            model_file (str): Path to the CNN model file.
            data_file (str): Path to the dataset file.
            learning_rate (float): Learning rate for the CNN training.
            generate_new_split (bool): Whether to generate a new data split.
            sep (str): Separator used in the dataset file.
            decimal (str): Decimal point character used in the dataset file.
            batchsize (int): Size of the batches for training.
            batch_split (bool): Whether to split data into batches.
            interpolation (bool): Whether to apply interpolation to the data.

        Raises:
            FileNotFoundError: If the specified data folder does not exist.
        """
        if generate_new_split:
            # Generate random seeds for data splitting.
            seed: list[int] = random.randint(0, 32000, N)
            
            for n in range(N):
                # Execute window splitting for each seed.
                WindowSplittingExecution.execute(
                    data_file, 
                    batch_split=batch_split, 
                    validation_split=True, 
                    test_size=test_size, 
                    seed=seed[n], 
                    batchsize=batchsize, 
                    interpolation=interpolation, 
                    window_size=10,
                    sep=sep, 
                    decimal=decimal
                )
        
        # Extract the dataset name from the file path.
        _, data_name = os.path.split(data_file)
        data_name, _ = os.path.splitext(data_name)

        # Define the folder path for the split data.
        folder = os.path.join('build', 'window_split', data_name)

        # List and sort the data files in descending order.
        data_files: list[str] = os.listdir(folder)
        data_files.sort(key=int, reverse=True)

        for n in range(N):
            # Execute CNN training for each split data file.
            CNNTrainingExecution.execute(
                model_file, 
                os.path.join(folder, data_files[n]), 
                'cnn.h5', 
                learning_rate
            )