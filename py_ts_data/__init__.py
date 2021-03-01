import os
from . import utils
import json

def path():
    this_file = __file__
    path_file_dir = os.path.dirname(os.path.dirname(this_file)) # PATH is parent dir
    with open(os.path.join(path_file_dir, "PATH"), "r") as f:
        return f.read().rstrip()

PATH = path()
print("TS-Data path: {}".format(PATH))

def data_info(name):
    """
    Returns JSON containing dataset information
    """
    dirname = os.path.join(PATH, name)
    with open(os.path.join(dirname, "meta"))as f:
        info = json.load(f)
    return info

def load_data(name, variables_as_channels=False):
    """
    Required arguments:
    name: Name of the dataset to load. Get the valid names from py_ts_data.list_datasets()

    Optional arguments:
    variables_as_channels (default False). If true, instead of shape = (x,y,z), the shape
    is given as (x,z,y).

    Returns tuple with 5 elements: X_train, y_train, X_test, y_train, info

    X_train and X_test return numpy arrays with shape: (x, y, z) where:
    x = number of timeseries in the dataset
    y = number of variables in each time series
    z = length of each series.

    If the dataset has variable lenght series, z = length of the longest series. Shorter
    series are filled with np.nan

    """

    info = data_info(name)
    train_file = os.path.join(PATH, name, "train")
    test_file = os.path.join(PATH, name, "test")

    X_train, y_train = utils.parse_file(train_file, info)
    X_test, y_test = utils.parse_file(test_file, info)

    if variables_as_channels:
        X_train = X_train.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)

    return X_train, y_train, X_test, y_test, info

def list_datasets():
    """
    Returns list of datasets available from py_ts_data.PATH
    """
    return os.listdir(PATH)
