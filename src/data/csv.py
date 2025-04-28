import os
from src.data.data_container import DataContainer
from src.data.data_type import DataType
from pandas import DataFrame, read_csv

class CSV(DataContainer):
    """
    Class for handling CSV files.
    
    Attributes:
        sep (str): Separator used in the CSV file.
        decimal (str): Decimal separator used in the CSV file.
        df (DataFrame): DataFrame containing the loaded CSV data.
        name (str): Name of the CSV file without extension.
    """
    def __init__(self, sep: str, decimal: str):
        self.sep: str = sep
        self.decimal: str = decimal
        self.df: DataFrame = None
        self.name: str = None

    @classmethod
    def from_file(cls, file: str, sep: str, decimal: str):
        """
        Create a CSV object from a file.
        
        Args:
            file (str): Path to the CSV file.
            sep (str): Separator used in the CSV file.
            decimal (str): Decimal separator used in the CSV file. 
        
        Returns:
            CSV: An instance of the CSV class with the loaded data.
        """
        csv = cls(sep, decimal)
        csv.load(file)
        csv.set_name(file)
        return csv
    
    @classmethod
    def from_df(cls, df: DataFrame, name: str, sep: str=',', decimal: str='.'):
        """
        Create a CSV object from a DataFrame.
        
        Args:
            df (DataFrame): DataFrame to be converted to CSV.
            name (str): Name of the CSV file without extension.
            sep (str): Separator used in the CSV file.
            decimal (str): Decimal separator used in the CSV file.
        
        Returns:
            CSV: An instance of the CSV class with the loaded data.
        """
        csv = cls(sep, decimal)
        csv.df = df
        csv.name = name
        return csv

    def load(self, file: str):
        """
        Load the CSV file into a DataFrame.
        
        Args:
            file (str): Path to the CSV file.    
        """
        self.df = read_csv(file, sep=self.sep, decimal=self.decimal)
    
    def save(self, folder: str):
        """
        Save the DataFrame to a CSV file.

        Args:
            folder (str): Path to the folder where the CSV file will be saved.
        """
        if not os.path.exists(folder): os.makedirs(folder)
        self.df.to_csv(os.path.join(folder, self.name + '.csv'))

    def set_name(self, file: str) -> None:
        """
        Set the name of the CSV file without extension.
        
        Args:
            file (str): Path to the CSV file.
        """
        _, tail = os.path.split(file)
        filename, _ = os.path.splitext(tail)
        self.name =  filename