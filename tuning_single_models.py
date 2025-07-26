import os
import sys
import importlib

# Projekt-Root hinzufügen, wenn nötig
#project_root = os.path.abspath('..')
#if project_root not in sys.path:
#    sys.path.insert(0, project_root)

import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import pandas as pd
from IPython import display  # Für Live-Update im Notebook, kann evtl. weg
from pandasgui import show
from tqdm import tqdm
import seaborn as sns
from pprint import pprint
import pickle

import src.execution
import src.model
import src.util

importlib.reload(src.execution)
importlib.reload(src.model)
importlib.reload(src.util)

from src.data import NPY
from src.execution import CNNValidationExecution, WindowSplittingExecution, CNNTuningExecution, TrAdaBoostR2TrainingExecution, CNNTrainingExecution
from src.model import CNN, build_model_for_TrAdaBoostR2, tradaBoostR2_setup
from src.util import timestamp

def main():
    SPLITTED_DATA_FOLDER = "build\\window_split\\sim_data_preprocessed\\1743966827\\"
    CNNTuningExecution.execute_tuning_three_model(SPLITTED_DATA_FOLDER, train_on_scled_labels=True)

if __name__ == "__main__":
    print("Hier habe wwas gestartet")
    main()
