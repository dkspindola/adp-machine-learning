import os 
from src.experiment import SoftStartExperiment

DATA_FOLDER = "build/window_split/real-data"
MODEL_FILE = "assets/models/trained/normal-syn/1744563228/soft-start.h5"

data_folders = sorted([f.path for f in os.scandir(DATA_FOLDER) if f.is_dir()])

for data_folder in data_folders:
    SoftStartExperiment.run(model_file=MODEL_FILE, 
                            data_folder=data_folder,
                            lr_factors=[0.1],
                            unfreezed_layers=[8])