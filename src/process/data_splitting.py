from typing import Tuple
from pandas import DataFrame

from src.data import CSV
from src.data import DataType
from src.process.splitting import Splitting

from lib.Splitting_Scaling_Function import Split_Scaling

class DataSplitting(Splitting):
    def __init__(self, batch_split, validation_split):
        super().__init__(batch_split, validation_split)

    def start(self, test_size: float=0.2, seed: int=42, batchsize: int=1800):
        self.splitted_data = []

        batch_split_number: int = 2 if self.batch_split else 1
        validation_split_number: int = 1 if self.validation_split else 0

        dfs: Tuple[DataFrame, ...] = Split_Scaling(self.data.df, test_size, seed, Validation_Data=validation_split_number, standard=batch_split_number, batchsize=batchsize,  save=False)
        self.set_splitted_data(dfs)

    def set_splitted_data(self, df: Tuple[DataFrame, ...]) -> None:
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
        timestamp: list[str] = []
        filename: list[str] = []
        shape: list[Tuple[int, int]] = []

        df: DataFrame = DataFrame()
        
        for key in self.splitted_data.keys():
            print(f'✔\tSave data to \'{self.filepaths[key]}\'')
            print(f'\tshape:\t{self.splitted_data[key].shape}')
            print(f'\tmemory:\t{self.splitted_data[key].memory_usage(deep=True, index=False).sum()} bytes')
            print()