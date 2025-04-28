from src.execution import CNNTrainingExecution
from src.experiment import MultipleCNNTrainingExperiment
import os
import json

DATA_FILE = "assets/data/real-data.csv"
MODEL_FILE = "assets/models/untrained/real-data/1743693239/best_model.h5" # With file extension

for test_size in [0.95, 0.9, 0.85, 0.8, 0.5]:
    best_hp = json.load(open(f"{os.path.dirname(MODEL_FILE)}/best-hyperparameters.json", "r"))

    MultipleCNNTrainingExperiment.start(N=1, 
                                    test_size=test_size, 
                                    model_file=MODEL_FILE, 
                                    data_file=DATA_FILE,
                                    learning_rate=best_hp["learning_rate"], 
                                    generate_new_split=True,
                                    sep=";",
                                    decimal=",", 
                                    batchsize=1800, 
                                    batch_split=True, 
                                    interpolation=False)