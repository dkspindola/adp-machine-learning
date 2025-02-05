from enum import Enum

class DataType(Enum):
    REAL_DATA = 'real-data'
    SYNTHETIC_DATA = 'synthetic-data'
    X_TRAIN = 'x-train'
    X_VALIDATE = 'x-validate'
    X_TEST = 'x-test'
    Y_TRAIN = 'y-train'
    Y_VALIDATE = 'y-validate'
    Y_TEST = 'y-test'
    X_TRAIN_SCALED = 'x-train-scaled'
    X_VALIDATE_SCALED = 'x-validate-scaled'
    X_TEST_SCALED = 'x-test-scaled'
    Y_TRAIN_SCALED = 'y-train-scaled'
    Y_VALIDATE_SCALED = 'y-validate-scaled'
    Y_TEST_SCALED = 'y-test-scaled'
    X_TEST_SCALED_INTERPOLATED = 'x-test-scaled-interpolated'
    Y_TEST_INTERPOLATED = 'y-test-interpolated'
