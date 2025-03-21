import os
from src.data.data_container import DataContainer
from src.data.data_type import DataType
from pandas import DataFrame, read_csv

class CSV(DataContainer):
    def __init__(self, sep: str, decimal: str):
        self.sep: str = sep
        self.decimal: str = decimal
        self.df: DataFrame = None
        self.name: str = None

    @classmethod
    def from_file(cls, file: str, sep: str, decimal: str):
        csv = cls(sep, decimal)
        csv.load(file)
        csv.set_type(file)
        return csv
    
    @classmethod
    def from_df(cls, df: DataFrame, name: str, sep: str=',', decimal: str='.'):
        csv = cls(sep, decimal)
        csv.df = df
        csv.name = name
        return csv

    def load(self, file: str):
        self.df = read_csv(file, sep=self.sep, decimal=self.decimal)
    
    def save(self, folder: str):
        if not os.path.exists(folder): os.makedirs(folder)
        self.df.to_csv(os.path.join(folder, self.name + '.csv'))

    def set_type(self, file: str) -> None:
        _, tail = os.path.split(file)
        filename, _ = os.path.splitext(tail)
        self.name =  filename