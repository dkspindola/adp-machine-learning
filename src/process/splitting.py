import os

from src.data import CSV
from src.serialize import Serializable
from src.process.process import Process

class Splitting(Process, Serializable):
    def __init__(self, batch_split: bool, validation_split: bool) -> None:
        self.batch_split = batch_split
        self.validation_split = validation_split
        self.data: CSV = None
        self.splitted_data: list[CSV] = None

    def load(self, file: str, sep: str=';', decimal:str=','):
        self.data = CSV.from_file(file, sep, decimal)
    
    def save(self, folder: str):
        os.makedirs(folder)
        for data in self.splitted_data:
            data.save(folder)

    def start(self):
        ... 
