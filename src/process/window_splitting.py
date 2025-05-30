import os
from numpy import ndarray
from typing import Tuple
from pandas import DataFrame
from src.data import NPY, DataType
from src.process.splitting import Splitting
from lib.Fensterung_Scaling_DeepLearning import Fensterung_Scale


class WindowSplitting(Splitting):
    """Handles window-based splitting of data for machine learning preprocessing.

    This class extends the `Splitting` class and provides functionality for
    splitting data into training, validation, and test sets with optional
    interpolation and batch splitting.
    """

    def __init__(self, batch_split: bool, validation_split: bool, interpolation: bool):
        """Initializes the WindowSplitting instance.

        Args:
            batch_split (bool): Whether to perform batch splitting.
            validation_split (bool): Whether to include a validation split.
            interpolation (bool): Whether to apply interpolation.
        """
        super().__init__(batch_split, validation_split)
        self.interpolation = interpolation

    def start(self, test_size: float = 0.2, seed=42, batchsize: int = 1800, window_size: int = 10):
        """Starts the window splitting process.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            seed (int): Random seed for reproducibility.
            batchsize (int): Size of each batch for splitting.
            window_size (int): Size of the window for splitting.

        Returns:
            None
        """
        self.splitted_data = []

        batch_split_number: int = 2 if self.batch_split else 1
        validation_split_number: int = 1 if self.validation_split else 0
        interpolation_number = 1 if self.interpolation else 0

        array: Tuple[ndarray, ...] = Fensterung_Scale(
            self.data.df,
            window_size=window_size,
            Datengröße=batchsize,
            size=test_size,
            Train_Test_Split=batch_split_number,
            Validation_data=validation_split_number,
            random=int(seed),
            Interpolation=interpolation_number,
        )
        self.set_splitted_data(array)
        self.add_scaler(array)

    def add_scaler(self, array: Tuple[ndarray, ...]):
        if self.validation_split and not self.interpolation:
            self.scalers_features =array[9]
            self.scalers_labels = array[10]    
        elif self.validation_split and self.interpolation and self.batch_split:
            self.scalers_features =array[9]
            self.scalers_labels = array[10]    
        elif self.validation_split and self.interpolation and not self.batch_split:
            self.scalers_features =array[6]
            self.scalers_labels = array[7]        
        elif not self.validation_split and self.interpolation:
            self.scalers_features =array[6]
            self.scalers_labels = array[7]    
        else:
            self.scalers_features =array[6]
            self.scalers_labels = array[7]    
            
            
    def get_scalers(self):
        return self.scalers_features, self.scalers_labels
    


    def set_splitted_data(self, array: Tuple[ndarray, ...]):
        """Sets the splitted data based on the provided array.

        Args:
            array (Tuple[ndarray, ...]): The array containing the split data.

        Returns:
            None
        """
        if self.validation_split and not self.interpolation:
            self.splitted_data.append(NPY.from_array(array[0], DataType.X_TRAIN_SCALED))
            self.splitted_data.append(NPY.from_array(array[1], DataType.X_VALIDATE_SCALED))
            self.splitted_data.append(NPY.from_array(array[2], DataType.X_TEST_SCALED))
            self.splitted_data.append(NPY.from_array(array[3], DataType.Y_TRAIN_SCALED))
            self.splitted_data.append(NPY.from_array(array[4], DataType.Y_VALIDATE_SCALED))
            self.splitted_data.append(NPY.from_array(array[5], DataType.Y_TEST_SCALED))
            self.splitted_data.append(NPY.from_array(array[6], DataType.Y_TRAIN))
            self.splitted_data.append(NPY.from_array(array[7], DataType.Y_VALIDATE))
            self.splitted_data.append(NPY.from_array(array[8], DataType.Y_TEST))

        elif self.validation_split and self.interpolation and self.batch_split:
            self.splitted_data.append(NPY.from_array(array[0], DataType.X_TRAIN_SCALED))
            self.splitted_data.append(NPY.from_array(array[1], DataType.X_VALIDATE_SCALED))
            self.splitted_data.append(NPY.from_array(array[2], DataType.X_TEST_SCALED))
            self.splitted_data.append(NPY.from_array(array[3], DataType.Y_TRAIN_SCALED))
            self.splitted_data.append(NPY.from_array(array[4], DataType.Y_VALIDATE_SCALED))
            self.splitted_data.append(NPY.from_array(array[5], DataType.Y_TEST_SCALED))
            self.splitted_data.append(NPY.from_array(array[6], DataType.Y_TRAIN))
            self.splitted_data.append(NPY.from_array(array[7], DataType.Y_VALIDATE))
            self.splitted_data.append(NPY.from_array(array[8], DataType.Y_TEST))
            self.splitted_data.append(NPY.from_array(array[11], DataType.X_TEST_SCALED_INTERPOLATED))
            self.splitted_data.append(NPY.from_array(array[12], DataType.Y_TEST_INTERPOLATED))

        elif self.validation_split and self.interpolation and not self.batch_split:
            self.splitted_data.append(NPY.from_array(array[0], DataType.X_TRAIN_SCALED))
            self.splitted_data.append(NPY.from_array(array[1], DataType.X_VALIDATE_SCALED))
            self.splitted_data.append(NPY.from_array(array[2], DataType.X_TEST_SCALED))
            self.splitted_data.append(NPY.from_array(array[3], DataType.Y_TRAIN_SCALED))
            self.splitted_data.append(NPY.from_array(array[4], DataType.Y_VALIDATE_SCALED))
            self.splitted_data.append(NPY.from_array(array[5], DataType.Y_TEST_SCALED))
            self.splitted_data.append(NPY.from_array(array[6], DataType.Y_TRAIN))
            self.splitted_data.append(NPY.from_array(array[7], DataType.Y_VALIDATE))
            self.splitted_data.append(NPY.from_array(array[8], DataType.Y_TEST))

        elif not self.validation_split and self.interpolation:
            self.splitted_data.append(NPY.from_array(array[0], DataType.X_TRAIN_SCALED))
            self.splitted_data.append(NPY.from_array(array[1], DataType.X_TEST_SCALED))
            self.splitted_data.append(NPY.from_array(array[2], DataType.Y_TRAIN_SCALED))
            self.splitted_data.append(NPY.from_array(array[3], DataType.Y_TEST_SCALED))
            self.splitted_data.append(NPY.from_array(array[4], DataType.Y_TRAIN))
            self.splitted_data.append(NPY.from_array(array[5], DataType.Y_TEST))
            self.splitted_data.append(NPY.from_array(array[8], DataType.X_TEST_SCALED_INTERPOLATED))
            self.splitted_data.append(NPY.from_array(array[9], DataType.Y_TEST_INTERPOLATED))

        else:
            self.splitted_data.append(NPY.from_array(array[0], DataType.X_TRAIN_SCALED))
            self.splitted_data.append(NPY.from_array(array[1], DataType.X_TEST_SCALED))
            self.splitted_data.append(NPY.from_array(array[2], DataType.Y_TRAIN_SCALED))
            self.splitted_data.append(NPY.from_array(array[3], DataType.Y_TEST_SCALED))
            self.splitted_data.append(NPY.from_array(array[4], DataType.Y_TRAIN))
            self.splitted_data.append(NPY.from_array(array[5], DataType.Y_TEST))