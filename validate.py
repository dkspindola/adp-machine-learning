import os
from src.experiment import MultipleCNNValidationExperiment
from src.show import plot_mean_with_uncertainty

N = 10
MAKE_VALIDATION = True

folders = ["build/soft-start/soft-start/5", "build/soft-start/soft-start/10", "build/soft-start/soft-start/15", 
           "build/soft-start/soft-start/20", "build/soft-start/soft-start/50"]

for folder in folders:
    MultipleCNNValidationExperiment.start(folder, N, MAKE_VALIDATION, None)
