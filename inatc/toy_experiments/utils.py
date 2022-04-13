import yaml
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


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
    X, y = make_classification(**kwargs)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test