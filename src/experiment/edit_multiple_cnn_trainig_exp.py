import os
import random
from console import Console

class Observer:
    def update(self, message: str):
        pass

class Logger(Observer):
    def __init__(self):
        self.console = Console()

    def update(self, message: str):
        self.console.log(message)

class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer: Observer):
        self._observers.append(observer)

    def detach(self, observer: Observer):
        self._observers.remove(observer)

    def notify(self, message: str):
        for observer in self._observers:
            observer.update(message)

class MultipleCNNTrainingExperiment(Subject):
    @classmethod
    def start(cls, N: int, test_size: float, model_file: str, data_file: str):
        experiment = cls()
        logger = Logger()
        experiment.attach(logger)

        experiment.notify("Experiment started")
        seed: list[int] = random.sample(range(100), N)
        experiment.notify(f"Generated seeds: {seed}")

        '''
        for n in range(N):
            WindowSplittingExecution.execute(data_file, 
                                             batch_split= False, 
                                             validation_split=True, 
                                             test_size=test_size, 
                                             seed=seed[n], 
                                             batchsize=1800, 
                                             interpolation=False, 
                                             window_size=10)
        '''
        data_file: list[str] = os.listdir('build/split')
        data_file.sort(key=int, reverse=True)
        experiment.notify(f"Data files sorted: {data_file}")

        experiment.notify("Experiment finished")