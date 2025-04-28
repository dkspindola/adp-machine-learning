import os
from src.experiment import MultipleCNNValidationExperiment
from src.show import plot_mean_with_uncertainty

N = 15
MAKE_VALIDATION = True

folders = ["build/train/best-model-1-5", "build/train/best-model-1-10", "build/train/best-model-1-15", 
           "build/train/best-model-1-20", "build/train/best-model-1-50"]

for folder in folders:
    MultipleCNNValidationExperiment.start(folder, N, MAKE_VALIDATION, None)
