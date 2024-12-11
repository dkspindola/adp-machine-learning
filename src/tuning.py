import os
import json
import joblib
import time

import numpy as np

import json

from pandas import DataFrame

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

class Tunable():
    def tune() -> None: ...


class RandomForest(Tunable):
    def __init__(self, config: str):
        self.config: str = config

        self.timestamp: int = None
        self.param_distributions: dict[str, list] = None
        self.n_iter: int = None
        self.cv: int = None

    def load(self) -> None:
        print(f'\tLoad file \'{self.config}\'', end='\r')

        file = open(self.config, 'r')
        config: dict[str] = json.load(file)
        file.close()

        self.param_distributions = config['param_distributions']
        self.n_iter = config['n_iter']
        self.cv = config['cv']

        print('✔')


        

    def tune(self, x_train: DataFrame, y_train: DataFrame, verbose: int, n_jobs: int, seed: int, return_train_score: bool, savepath: str) -> None:
        if (self.param_distributions is None): self.load()

        # Random search of parameters
        cv_result = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=self.param_distributions, n_iter=self.n_iter, cv=self.cv, verbose=verbose, random_state=seed, n_jobs=n_jobs, return_train_score=return_train_score)

        # Model fitting
        cv_result.fit(x_train, y_train)
        self.timestamp = int(time.time())

        # Print results
        #print(json.dumps(cv_result.best_params_))
        
        # Create unique folder
        folder: str = os.path.join(savepath, f'{self.timestamp}')
        os.makedirs(folder)

        # Save best parameters 
        file = open(os.path.join(folder, 'random-forest-best-params.json'), 'w')
        json.dump(cv_result.best_params_, file, indent=4)
        file.close()

        # Save best estimator
        joblib.dump(cv_result.best_estimator_, os.path.join(folder, 'random-forest-best-estimator.pkl'))