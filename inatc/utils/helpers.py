import argparse
import yaml
import urllib
import torch
import multiprocessing

import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def parse_arguments():
    """
    Creates an argument parser for balanced and unbalanced experiments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="Path to config file.", type=str)

    parser.add_argument(
        "-nc",
        "--neat_config",
        help="Path to NEAT config file.",
        type=str,
    )

    parser.add_argument(
        "--train",
        help="Path to training data.",
        type=str,
    )

    parser.add_argument(
        "--test",
        help="Path to testing data.",
        type=str,
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


def get_embeddings(text, tokenizer, model, embedding_layer="pooler_output"):
    """
    Obtains sentence embeddings from BERT.

    Arguments:
        text: Input text for which embeddings will be computed.
        tokenizer: BERTTokenizer instance.
        model: BERTModel instance.
    """

    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
    outputs = model(**inputs)
    return outputs[embedding_layer]


def is_internet():
    """
    Check to see if internet is available.
    """

    try:
        urllib.request.urlopen("http://google.com")
        return True
    except:
        return False


def sigmoid(z):
    """
    Perform sigmoid activation.

    Arguments:
        z: Array of inputs.
    """
    z = np.array(z)
    out = 1 / (1 + np.exp(-z))
    return list(out)


def label_transform(inputs, threshold=0.5):
    """
    Transform values into multi-label format based on its probabilities.

    Arguments:
        inputs: Array of probabilities for a particular label.
    """
    input_arr = np.array(inputs)
    input_arr[input_arr > threshold] = 1
    input_arr[input_arr != 1] = 0
    return list(input_arr)


def get_accelerator():
    """
    Returns accelerator and devices based on availability.
    """
    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"
    # Always return devices as 1 since distributed computing does not work.
    return accelerator, 1
