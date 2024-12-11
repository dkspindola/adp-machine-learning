from pandas import DataFrame
from pandas import read_csv

class Datacontainer:
    def __init__(self, path: str, batchsize: int, sep: str, decimal: str) -> None:
        self.path = path
        self.batchsize = batchsize
        self.sep = sep
        self.decimal = decimal
        self.data: DataFrame = None

    def load(self) -> None:
        self.data = read_csv(self.path, sep=self.sep, decimal=self.decimal)

    def print(self) -> None:
        print(self.data.info)