import argparse
from src.execution import CNNTuningExecution, DataSplittingExecution, WindowSplittingExecution, CNNTrainingExecution, CNNValidationExecution
from src.experiment import MultipleCNNTrainingExperiment, MultipleCNNValidationExperiment

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
    tune_parser.add_argument('--timestamp', default=None, type=int)

    #VALIDATE
    validate_parser = subparsers.add_parser('validate', help='Model validation')
    validate_parser.add_argument('--model_file', type=str)
    validate_parser.add_argument('--data_folder', type=str)

    #TRAIN
    train_parser = subparsers.add_parser('train', help='Model training')
    train_parser.add_argument('--model_file', type=str)
    train_parser.add_argument('--data_folder', type=str)

    #EXPERIMENT
    mul_cnn_train = subparsers.add_parser('mul_cnn_exp')
    mul_cnn_train.add_argument('--N', default=10, type=int)
    mul_cnn_train.add_argument('--test_size', default=0.2, type=float)
    mul_cnn_train.add_argument('--model_file', type=str)
    mul_cnn_train.add_argument('--data_file', type=str)
    mul_cnn_train.add_argument('--learning_rate', type=float)
    mul_cnn_train.add_argument('--generate_new_split', action='store_true', default=False)

    mul_cnn_val = subparsers.add_parser('mul_cnn_val')
    mul_cnn_val.add_argument('--model_folder', type=str)    
    mul_cnn_val.add_argument('--data_folder', type=str) 
    mul_cnn_val.add_argument('--N', type=int, default=10)

    args = parser.parse_args()

    function: dict = {'split': lambda: DataSplittingExecution.execute(args.data, args.batch_split, args.validation_split, args.test_size, args.seed, args.batchsize),
                      'tune': lambda: CNNTuningExecution.execute(args.data_folder, ts=args.timestamp),
                      'window_split': lambda: WindowSplittingExecution.execute(args.data, args.batch_split, args.validation_split, args.test_size, args.seed, args.batchsize, args.interpolation, args.window_size),
                      'train': lambda: CNNTrainingExecution.execute(args.model_file, args.data_folder),
                      'validate': lambda: CNNValidationExecution.execute(args.model_file, args.data_folder),
                      'mul_cnn_val': lambda: MultipleCNNValidationExperiment.start(args.model_folder, args.data_folder, args.N),
                      'mul_cnn_exp': lambda: MultipleCNNTrainingExperiment.start(args.N, args.test_size, args.model_file, args.data_file, args.learning_rate, args.generate_new_split)}
    
    function[args.command]()


if __name__ == "__main__":
    main()
