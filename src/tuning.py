import os
import json
import joblib
import time

import numpy as np

import json

from pandas import DataFrame

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

from lib.funcions_konventionelle_Modelle.Hyperparametertuning_Konventionell import validate_best_model_random_forest

class Tunable():
    def tune() -> None: ...


class RandomForest(Tunable):
    CONFIG_PATH = 'config/tune/random-forest/{}.json'
    MODEL_PATH = 'build/tune/random-forest/{}/model.pkl'
    BEST_PARAMS_PATH = 'build/tune/random-forest/{}/best-params.json'

    def __init__(self, config_name: str=None, model_id: int=None):
        self.config_name: str = config_name
        self.timestamp: int = model_id

        self.config_path: str = None
        self.model_path: str = None
        self.best_params_path: str = None

        self.param_distributions: dict[str, list] = None
        self.n_iter: int = None
        self.cv: int = None

        self.model: dict = None

    def save(self) -> None:
        # Create folder
        self.best_params_path = RandomForest.BEST_PARAMS_PATH.format(self.timestamp)
        self.model_path = RandomForest.MODEL_PATH.format(self.timestamp)

        os.makedirs(os.path.dirname(self.best_params_path))

        # Save best parameters 
        file = open(self.best_params_path, 'w')
        json.dump(self.model.best_params_, file, indent=4)
        file.close()

        # Save best estimator
        joblib.dump(self.model, self.model_path)
        

    def load(self) -> None:
        if (self.config_name is not None):  
            self.config_path = RandomForest.CONFIG_PATH.format(self.config_name)
            print(f'\tLoad config \'{self.config_path}\'', end='\r')
            self.load_config()
            print('✔')

        if (self.timestamp is not None): 
            self.model_path = RandomForest.MODEL_PATH.format(self.timestamp)
            print(f'\tLoad model \'{self.model_path}\'', end='\r')
            self.load_model()
            print('✔')

        print()

    def load_model(self) -> None:
        self.model: dict = joblib.load(self.model_path)


    def load_config(self) -> None:
        file = open(self.config_path, 'r')
        config: dict[str] = json.load(file)
        file.close()

        self.param_distributions = config['param_distributions']
        self.n_iter = config['n_iter']
        self.cv = config['cv']

    def tune(self, x_train: DataFrame, y_train: DataFrame, verbose: int, n_jobs: int, seed: int, return_train_score: bool) -> None:
        if (self.param_distributions is None): self.load()

        # Random search of parameters
        cv_result = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=self.param_distributions, n_iter=self.n_iter, cv=self.cv, verbose=verbose, random_state=seed, n_jobs=n_jobs, return_train_score=return_train_score)

        # Model fitting
        cv_result.fit(x_train, y_train)

        # 
        self.timestamp = int(time.time())
        self.model = cv_result

        self.save()
        

    def validate(self, x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame) -> None:
        validate_best_model_random_forest(self.model, x_train, x_test, y_train, y_test)