from src.execution import CNNTrainingExecution
from src.experiment import MultipleCNNTrainingExperiment
import os
import json

DATA_FOLDER = "build/window_split/real-data/1744034830"
MODEL_FILE = "build/tune/tune_test/1744495142/best-model-1.h5" # With file extension



data_folders = sorted([f.path for f in os.scandir("build/tune/tune_test") if f.is_dir()])
model_folders = sorted([f.path for f in os.scandir("build/tune/tune_test") if f.is_dir()])


for test_size in [0.2]:
    best_hp = json.load(open(f"{os.path.dirname(MODEL_FILE)}/best-hyperparameters.json", "r"))

    MultipleCNNTrainingExperiment.start(N=5, 
                                    test_size=test_size, 
                                    model_file=MODEL_FILE, 
                                    data_file="assets/data/real-data.csv",
                                    learning_rate=best_hp["learning_rate"], 
                                    generate_new_split=True,
                                    sep=";",
                                    decimal=",", 
                                    batchsize=1800, 
                                    batch_split=True, 
                                    interpolation=False)