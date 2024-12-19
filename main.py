import os
import argparse
import pandas as pd
from pandas import DataFrame
from src.datacontainer import Datacontainer
from src.model import train

from src.splitting import split
from src.tuning import tune, validate

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    #test_parser = subparsers.add_parser("test", help="Test the model")

    # SPLIT
    split_parser = subparsers.add_parser("split", help='Split data')
    split_parser.add_argument("--data", help="Path to load the data from")
    split_parser.add_argument("--batchsize", default=1800, help="Number of datapoints per experiment", type=int)
    split_parser.add_argument("--test_size", default=0.2, help="Path to load the data from", type=float)
    split_parser.add_argument('--validate', action='store_true', help='Whether or not to split test data')
    split_parser.add_argument('--batch_split', action='store_true', help='Respect the batches while splitting')
    split_parser.add_argument('--save', default='build/split', help='Path to save splitted data to')
    split_parser.add_argument('--seed', default=42, help='Seed for random number generation')

    #TUNE
    tune_parser = subparsers.add_parser('tune', help='Hyperparameter tuning')
    tune_parser.add_argument('--data', help='Timestamp of test data', type=int)
    tune_parser.add_argument('--config', default=None, help='Name of config file', type=str)
    tune_parser.add_argument('--verbose', default=3, help='Number of messages during process', type=int)
    tune_parser.add_argument('--n_jobs', default=2, help='Number cores to use for computing (-1 to use all)', type=int)
    tune_parser.add_argument('--seed', default=42, help='Seed for random number generation')
    tune_parser.add_argument('--return_train_score', default=True, help='Print train score at the end of tuning', type=bool)

    #VALIDATE
    validate_parser = subparsers.add_parser('validate', help='Model validation')
    validate_parser.add_argument('--model', help='Timestamp of model', type=int)
    validate_parser.add_argument('--data', help='Timestamp of train/test data', type=int)

    #TRAIN
    train_parser = subparsers.add_parser("train", help="Train CNN model")
    #train_parser.add_argument('--windowing', action='store_true', help='Do windowing while splitting')
    #train_parser.add_argument('--window_size', default=10, help='Size of window', type=int)

    args = parser.parse_args()

    function: dict = {'split': lambda: split(args.data, args.test_size, args.validate, args.batchsize, args.batch_split, args.windowing, args.window_size, args.save, args.seed),
                      'tune': lambda: tune(args.data, args.config, args.verbose, args.n_jobs, args.seed, args.return_train_score),
                      'validate': lambda: validate(args.model, args.data), 
                      'train': lambda: train()}
    
    function[args.command]()


if __name__ == "__main__":
    main()
    print('\tDone!')
