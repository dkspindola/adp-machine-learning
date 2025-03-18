import os
import numpy as np
from numpy import ndarray
from src.data.data_container import DataContainer
from src.data.data_type import DataType

class NPY(DataContainer):
    def __init__(self):
        self.array: ndarray = None
        self.type: DataType = None

    @classmethod
    def from_file(cls, file: str):
        npy = cls()
        npy.load(file)
        npy.set_type(file)
        return npy
    
    @classmethod
    def from_array(cls, array: ndarray, data_type: DataType):
        npy = cls()
        npy.array = array
        npy.type = data_type
        return npy

    def load(self, file: str):
        self.array = np.load(file)   
    
    def save(self, folder: str):
        if not os.path.exists(folder): os.makedirs(folder)
        np.save(os.path.join(folder, self.type.value + '.npy'), self.array)

    def set_type(self, file: str) -> None:
        _, tail = os.path.split(file)
        filename, _ = os.path.splitext(tail)
        self.type =  DataType(filename)