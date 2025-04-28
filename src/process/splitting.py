import os

from src.data import CSV
from src.serialize import Serializable
from src.process.process import Process

class Splitting(Process, Serializable):
    """Handles splitting of data into batches and validation sets.

    This class provides functionality to load data, split it into
    batches or validation sets, and save the resulting splits.

    Attributes:
        batch_split (bool): Whether to perform batch splitting.
        validation_split (bool): Whether to perform validation splitting.
        data (CSV): The loaded data.
        splitted_data (list[CSV]): The resulting split data.
    """

    def __init__(self, batch_split: bool, validation_split: bool) -> None:
        """Initializes the Splitting class.

        Args:
            batch_split (bool): Whether to perform batch splitting.
            validation_split (bool): Whether to perform validation splitting.
        """
        self.batch_split = batch_split
        self.validation_split = validation_split
        self.data: CSV = None
        self.splitted_data: list[CSV] = None

    def load(self, file: str, sep, decimal):
        """Loads data from a file.

        Args:
            file (str): The path to the file to load.
            sep: The delimiter used in the file.
            decimal: The decimal separator used in the file.
        """
        self.data = CSV.from_file(file, sep, decimal)
    
    def save(self, folder: str):
        """Saves the split data to a specified folder.

        Args:
            folder (str): The path to the folder where data will be saved.
        """
        os.makedirs(folder)
        for data in self.splitted_data:
            data.save(folder)

    def start(self):
        """Starts the splitting process.

        This method should be implemented to define the logic for
        splitting the data into batches or validation sets.
        """
        ...
