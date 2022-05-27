import argparse
import yaml
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


def parse_arguments():
    """
    Creates an argument parser for balanced and unbalanced experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="Name of the dataset config file.", type=str
    )
    parser.add_argument(
        "-n", "--run_name", help="The name of the current run.", type=str
    )
    parser.add_argument(
        "-s", "--seed", help="Seed for random number generator.", type=int
    )
    return parser.parse_args()


def read_yaml(filename):
    """
    Reads yaml configuration file.

    Arguments:
        filename: The name of the yaml file.
    """
    with open(filename, "r") as yfile:
        cfg = yaml.load(yfile, Loader=yaml.FullLoader)
    return cfg


def read_fake_data(split_size, random_state, **kwargs):
    """
    Creates and returns fake train and test data.

    Arguments:
        split_size: Training and testing split size.
        random_state: Random state seed for the dataset.
    """
    X, y = make_classification(random_state=random_state, **kwargs)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test