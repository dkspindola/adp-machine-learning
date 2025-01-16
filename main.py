import argparse
from src.execution import CNNTuningExecution, DataSplittingExecution, WindowSplittingExecution

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # SPLIT
    split_parser = subparsers.add_parser("split", help='Split data')
    split_parser.add_argument("--data", help="Path to load the data from")
    split_parser.add_argument("--batchsize", default=1800, help="Number of datapoints per experiment", type=int)
    split_parser.add_argument("--test_size", default=0.2, help="Path to load the data from", type=float)
    split_parser.add_argument('--validation_split', action='store_true', help='Whether or not to split test data', default=True)
    split_parser.add_argument('--batch_split', action='store_true', help='Respect the batches while splitting', default=False)
    split_parser.add_argument('--seed', default=42, help='Seed for random number generation')
    split_parser.add_argument('--window_size', default=10, type=int)
    split_parser.add_argument('--interpolation', default=False, action='store_true')

    #TUNE
    tune_parser = subparsers.add_parser('tune', help='Hyperparameter tuning')
    tune_parser.add_argument('--data_folder', help='Folder of the data', type=str)

    #VALIDATE
    validate_parser = subparsers.add_parser('validate', help='Model validation')
    validate_parser.add_argument('--model', help='Timestamp of model', type=int)
    validate_parser.add_argument('--data', help='Timestamp of train/test data', type=int)

    args = parser.parse_args()

    function: dict = {'split': lambda: DataSplittingExecution.execute(args.data, args.batch_split, args.validation_split, args.test_size, args.seed, args.batchsize),
                      'tune': lambda: CNNTuningExecution.execute(args.data_folder),
                      'window_split': lambda: WindowSplittingExecution.execute(args.data, args.batch_split, args.validation_split, args.test_size, args.seed, args.batchsize, args.interpolation, args.window_size)
                      }
    
    function[args.command]()


if __name__ == "__main__":
    main()
    print('\tDone!')
