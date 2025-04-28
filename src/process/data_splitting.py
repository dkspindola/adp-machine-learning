from typing import Tuple
from pandas import DataFrame

from src.data import CSV
from src.data import DataType
from src.process.splitting import Splitting

from lib.Splitting_Scaling_Function import Split_Scaling

class DataSplitting(Splitting):
    """Handles data splitting and scaling for machine learning workflows.

    Attributes:
        batch_split (bool): Indicates whether batch splitting is enabled.
        validation_split (bool): Indicates whether validation splitting is enabled.
        splitted_data (list): Stores the split data as CSV objects.
    """

    def __init__(self, batch_split: bool, validation_split: bool):
        """Initializes the DataSplitting class.

        Args:
            batch_split (bool): Whether to enable batch splitting.
            validation_split (bool): Whether to enable validation splitting.
        """
        super().__init__(batch_split, validation_split)

    def start(self, test_size: float = 0.2, seed: int = 42, batchsize: int = 1800):
        """Splits the data into training, validation, and test sets.

        Args:
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            batchsize (int, optional): Size of each batch. Defaults to 1800.
        """
        self.splitted_data = []

        batch_split_number: int = 2 if self.batch_split else 1
        validation_split_number: int = 1 if self.validation_split else 0

        dfs: Tuple[DataFrame, ...] = Split_Scaling(
            self.data.df, test_size, seed, Validation_Data=validation_split_number,
            standard=batch_split_number, batchsize=batchsize, save=False
        )
        self.set_splitted_data(dfs)

    def set_splitted_data(self, df: Tuple[DataFrame, ...]) -> None:
        """Stores the split data into the `splitted_data` attribute.

        Args:
            df (Tuple[DataFrame, ...]): Tuple of DataFrames containing split data.
        """
        if self.validation_split:
            self.splitted_data.append(CSV.from_df(df[0], DataType.X_TRAIN.value))   
            self.splitted_data.append(CSV.from_df(df[1], DataType.X_VALIDATE.value))     
            self.splitted_data.append(CSV.from_df(df[2], DataType.X_TEST.value))     
            self.splitted_data.append(CSV.from_df(df[3], DataType.Y_TRAIN.value))
            self.splitted_data.append(CSV.from_df(df[4], DataType.Y_VALIDATE.value))     
            self.splitted_data.append(CSV.from_df(df[5], DataType.Y_TEST.value))     
        else:                   
            self.splitted_data.append(CSV.from_df(df[0], DataType.X_TRAIN.value))
            self.splitted_data.append(CSV.from_df(df[0], DataType.X_TEST.value))   
            self.splitted_data.append(CSV.from_df(df[0], DataType.Y_TRAIN.value))   
            self.splitted_data.append(CSV.from_df(df[0], DataType.Y_TEST.value))   

    def print_summary(self):
        """Prints a summary of the split data, including file paths, shapes, and memory usage."""
        timestamp: list[str] = []
        filename: list[str] = []
        shape: list[Tuple[int, int]] = []

        df: DataFrame = DataFrame()
        
        for key in self.splitted_data.keys():
            print(f'✔\tSave data to \'{self.filepaths[key]}\'')
            print(f'\tshape:\t{self.splitted_data[key].shape}')
            print(f'\tmemory:\t{self.splitted_data[key].memory_usage(deep=True, index=False).sum()} bytes')
            print()