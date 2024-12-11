import os
import time

from typing import Tuple

from tabulate import tabulate

import pandas as pd
from pandas import DataFrame

from lib.Splitting_Scaling_Function import Split_Scaling
from lib.Fensterung_Scaling_DeepLearning import Fensterung_Scale

from src.datacontainer import Datacontainer

class Splitable:
    def split(self) -> None:    ...

class Splitting(Splitable, Datacontainer):
    def __init__(self, loadpath: str, batchsize: int, sep: str, decimal: str) -> None:
        self.splitted_data: dict[str, DataFrame] = None
        self.timestamp: int = None
        super().__init__(loadpath, batchsize, sep, decimal)

    def split(self, test_size: float, validation_split: bool, batch_split: bool, seed: int) -> None:
        if self.data is None: self.load()

        self.timestamp = int(time.time())

        batch_split_number: int = 2 if batch_split else 1
        validation_split_number: int = 1 if validation_split else 0

        splitted_data: Tuple[DataFrame, ...] = Split_Scaling(self.data, test_size, seed, Validation_Data=validation_split_number, standard=batch_split_number, batchsize=self.batchsize,  save=False)
        keys: list[str] = self.keylist(validation_split)
        
        self.splitted_data =  dict(zip(keys, splitted_data))

        self.splitted_data

    def save(self, path: str) -> None:
        folder: str = os.path.join(path, f'{self.timestamp}')
        os.makedirs(folder)

        for key in self.splitted_data.keys():
            data: DataFrame = self.splitted_data[key]
            filepath: str = os.path.join(folder, f'{key}.csv')
            data.to_csv(filepath)

    def keylist(self, validation_split: bool) -> list[str]:
        if validation_split:    return ['x-train', 'x-validate', 'x-test', 'y-train', 'y-validate', 'y-test']
        else:                   return ['x-train', 'x-test', 'y-train', 'y-test']

    def print_summary(self):
        timestamp: list[str] = []
        filename: list[str] = []
        shape: list[Tuple[int, int]] = []

        df: DataFrame = DataFrame()
        
        for key in self.splitted_data.keys():
            print(f'{self.splitted_data[key].info(verbose=False, show_counts=False)}')

class Windowing(Splitable, Datacontainer):
    def __init__(self, path: str, batchsize: int, sep: str, decimal: str) -> None:
        self.splitted_data: dict[str, DataFrame] = None
        super().__init__(path, batchsize, sep, decimal)

    def split(self, test_size: float, validation_split: bool, batch_split: bool, window_size: int,  seed: int) -> None:
        if self.data is None: self.load()
        
        self.timestamp = int(time.time())
        
        batch_split_number: int = 2 if batch_split else 1
        validation_split_number: int = 1 if validation_split else 0

        splitted_data: Tuple[DataFrame, ...] = Fensterung_Scale(self.data, window_size=window_size, Datengröße=self.batchsize, size=test_size, Train_Test_Split=batch_split_number, Validation_data=validation_split_number, random=seed)
        print(splitted_data)