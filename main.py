import os
import argparse
import pandas as pd
from pandas import DataFrame
from src.datacontainer import Datacontainer

from src.splitting import Splitting, Windowing
from src.tuning import RandomForest

def split(loadpath: str, test_size: int, validate: bool, batchsize: int, batch_split: bool, windowing: bool, window_size: int, savepath: str, seed: int):
    if windowing:
        splitable = Windowing(loadpath, batchsize, sep=';', decimal=',')
        splitable.split(test_size, validate, batch_split, window_size, seed)
    else:
        splitable = Splitting(loadpath, batchsize, sep=';', decimal=',')
        splitable.split(test_size, validate, batch_split, seed)
        splitable.save(savepath)
        splitable.print_summary()

def tune(timestamp: int, config: str, verbose: int, n_jobs: int, seed: int, return_train_score: bool, savepath: str) -> None:
    x_train: Datacontainer = Datacontainer(os.path.join('build/split', str(timestamp), 'x-train.csv'), batchsize=None, sep=',', decimal='.')
    y_train: Datacontainer = Datacontainer(os.path.join('build/split', str(timestamp), 'y-train.csv'), batchsize=None, sep=',', decimal='.')

    x_train.load()
    y_train.load()

    random_forest = RandomForest(os.path.join('config/tune/random-forest', f'{config}.json'))
    random_forest.tune(x_train.data, y_train.data, verbose, n_jobs, seed, return_train_score, savepath)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    #train_parser = subparsers.add_parser("train", help="Train the model")

    #test_parser = subparsers.add_parser("test", help="Test the model")

    # SPLIT
    split_parser = subparsers.add_parser("split", help='Split data')
    split_parser.add_argument("--load", help="Path to load the data from")
    split_parser.add_argument("--batchsize", default=1800, help="Number of datapoints per experiment", type=int)
    split_parser.add_argument("--test_size", default=0.2, help="Path to load the data from", type=float)
    split_parser.add_argument('--validate', action='store_true', help='Whether or not to split test data')
    split_parser.add_argument('--batch_split', action='store_true', help='Respect the batches while splitting')
    split_parser.add_argument('--windowing', action='store_true', help='Do windowing while splitting')
    split_parser.add_argument('--window_size', default=10, help='Size of window', type=int)
    split_parser.add_argument('--save', default='build/split', help='Path to save splitted data to')
    split_parser.add_argument('--seed', default=42, help='Seed for random number generation')

    #TUNE
    tune_parser = subparsers.add_parser('tune', help='Hyperparameter tuning')
    tune_parser.add_argument('--timestamp', help='Timestamp of test data', type=int)
    tune_parser.add_argument('--config', default='corvin', help='Name of config file', type=str)
    tune_parser.add_argument('--verbose', default=3, help='Number of messages during process', type=int)
    tune_parser.add_argument('--n_jobs', default=2, help='Number cores to use for computing (-1 to use all)', type=int)
    tune_parser.add_argument('--seed', default=42, help='Seed for random number generation')
    tune_parser.add_argument('--return_train_score', default=True, help='Print train score at the end of tuning', type=bool)
    tune_parser.add_argument('--save', default='build/tune', help='Path to save tuned data to')



    args = parser.parse_args()

    function: dict = {'split': lambda: split(args.load, args.test_size, args.validate, args.batchsize, args.batch_split, args.windowing, args.window_size, args.save, args.seed),
                      'tune': lambda: tune(args.timestamp, args.config, args.verbose, args.n_jobs, args.seed, args.return_train_score, args.save)}
    
    function[args.command]()


if __name__ == "__main__":
    main()
