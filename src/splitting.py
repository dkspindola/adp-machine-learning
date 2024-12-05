import os
import time

from tabulate import tabulate

import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler

from lib.Splitting_Scaling_Function import Split_Scaling

scaler_dict = {'standard': StandardScaler,
          'minmax': MinMaxScaler,
          'maxabs': MaxAbsScaler,
          'robust': RobustScaler
}

def split(data: DataFrame, test_size: float=0.2, scaler: str='standard', seed: int=0, validation: bool=True, batch_split: bool=False, batchsize: int=1800):
    split_type: int = 1 if not batch_split else 2

    if (validation):
        x_train, x_val, x_test, y_train, y_val, y_test = Split_Scaling(data, test_size, seed, scaler_dict[scaler], 1, split_type, batchsize, save=False)
        save(x_train, x_test, y_train, y_test, x_val, y_val)

    else:
        x_train, x_test, y_train, y_test = Split_Scaling(data, test_size, seed, scaler_dict[scaler], 0, split_type, batchsize, save=False)
        save(x_train, x_test, y_train, y_test)


def save(x_train: DataFrame, x_test: DataFrame, y_train: DataFrame, y_test: DataFrame, x_val: DataFrame=None, y_val: DataFrame=None):
    folder = f'build/split/{int(time.time())}'
    os.makedirs(folder)

    summarys = [] 

    x_train_path = os.path.join(folder, 'x-train.csv')
    x_train.to_csv(x_train_path)
    summarys.append(summary(x_train, x_train_path))

    if(x_val is not None):
        x_val_path: str = os.path.join(folder, 'x-validate.csv')
        x_val.to_csv(x_val_path)
        summarys.append(summary(x_val, x_val_path))


    x_test_path = os.path.join(folder, 'x-test.csv')
    x_test.to_csv(x_test_path)
    summarys.append(summary(x_test, x_test_path))

    y_train_path = os.path.join(folder, 'y-train.csv')
    y_train.to_csv(y_train_path)
    summarys.append(summary(y_train, y_train_path))

    if(y_val is not None):
        y_val_path: str = os.path.join(folder, 'y-validate.csv')
        y_val.to_csv(y_val_path)
        summarys.append(summary(y_val, y_val_path))

    
    y_test_path = os.path.join(folder, 'y-test.csv')
    y_test.to_csv(y_test_path)
    summarys.append(summary(y_test, y_test_path))

    print(tabulate(summarys, headers='keys', tablefmt='grid'))


def summary(data: DataFrame, path: str):
    return {'File': path, 'Size': len(data)}