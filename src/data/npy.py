import os
import numpy as np
from numpy import ndarray
from src.data.data_container import DataContainer
from src.data.data_type import DataType

class NPY(DataContainer):
    """Handles operations related to NPY files, including loading, saving, 
    and creating instances from files or arrays.
    """

    def __init__(self):
        """Initializes an NPY instance with an empty array and type."""
        self.array: ndarray = None
        self.type: DataType = None

    @classmethod
    def from_file(cls, file: str):
        """Creates an NPY instance from a file.

        Args:
            file (str): Path to the NPY file.

        Returns:
            NPY: An instance of the NPY class.
        """
        npy = cls()
        npy.load(file)
        npy.set_type(file)
        return npy
    
    @classmethod
    def from_array(cls, array: ndarray, data_type: DataType):
        """Creates an NPY instance from a numpy array and data type.

        Args:
            array (ndarray): The numpy array to store.
            data_type (DataType): The type of data represented by the array.

        Returns:
            NPY: An instance of the NPY class.
        """
        npy = cls()
        npy.array = array
        npy.type = data_type
        return npy

    def load(self, file: str):
        """Loads a numpy array from a file.

        Args:
            file (str): Path to the NPY file.
        """
        self.array = np.load(file)   
    
    def save(self, folder: str):
        """Saves the numpy array to a file in the specified folder.

        Args:
            folder (str): Path to the folder where the file will be saved.
        """
        if not os.path.exists(folder): os.makedirs(folder)
        np.save(os.path.join(folder, self.type.value + '.npy'), self.array)

    def set_type(self, file: str) -> None:
        """Sets the data type of the NPY instance based on the file name.

        Args:
            file (str): Path to the NPY file.
        """
        _, tail = os.path.split(file)
        filename, _ = os.path.splitext(tail)
        self.type =  DataType(filename)