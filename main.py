import argparse
import pandas as pd
from pandas import DataFrame

from src import splitting

def split(loadpath: str, validate: bool, test_size: float):
    data: DataFrame = pd.read_csv(loadpath, sep=';', decimal=',')
    splitting.split(data, validation=validate, test_size=test_size)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    #train_parser = subparsers.add_parser("train", help="Train the model")

    #test_parser = subparsers.add_parser("test", help="Test the model")

    split_parser = subparsers.add_parser("split", help='Split data')
    split_parser.add_argument("--load", help="Path to load the data from")
    split_parser.add_argument('--validate', action='store_true', help='Whether or not to split test data')
    split_parser.add_argument("--test_size", default=0.2, help="Path to load the data from", type=float)

    args = parser.parse_args()

    function: dict = {'split': lambda: split(args.load, args.validate, args.test_size)}
    
    function[args.command]()

if __name__ == "__main__":
    main()
