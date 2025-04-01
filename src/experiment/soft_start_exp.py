from src.model.cnn import CNN
from src.execution import CNNSoftStartExecution
from keras.optimizers import Adam

class SoftStartExperiment():
    @classmethod
    def run(cls, model_file: str, data_file: str, save_filename: str, learning_rates: list[float], ns_unfreezed_layers: list[int]):
        results = []
        for lr in learning_rates:
            for n in ns_unfreezed_layers:
                print(f"Testing with learning rate: {lr}, unfreezed layers: {n}")
                CNNSoftStartExecution.execute(model_file, data_file, save_filename, lr, n)
