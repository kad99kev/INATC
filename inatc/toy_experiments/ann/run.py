import os
import random
from model import SimpleModel
from inatc.toy_experiments.utils import read_fake_data, read_yaml, parse_arguments

from pytorch_lightning import LightningModule, Trainer

import torch
from torch.utils.data import TensorDataset, DataLoader


def prepare_data():
    """
    Parse arguments and prepare experiment data folders.
    """
    args = parse_arguments()

    # Create experiment directories based on the type of experiment.
    run_name = "runs/" + args.run_name + "/"
    if os.path.isdir(run_name):
        raise FileExistsError(
            "A previous run state already exists! Please rename the previous run or delete the folder."
        )
    else:
        os.mkdir(run_name)

    # Get config file based on type of experiment.
    data_file = f"configs/{args.config}"

    os.mkdir(run_name + "run_checkpoints")
    os.mkdir(run_name + "run_images")

    # Create fake dataset.
    cfg = read_yaml(data_file)
    split_size, random_state = cfg["info"].values()
    X_train, X_test, y_train, y_test = read_fake_data(
        split_size, random_state, **cfg["dataset"]
    )

    # Set seed.
    random.seed(random_state)

    return (
        cfg["training"],
        (cfg["dataset"]["n_features"], cfg["dataset"]["n_classes"]),
        (X_train, X_test, y_train, y_test),
        run_name,
    )


def run(training_info, model_info, dataset, run_name):
    """
    Train the Neural Network.

    Arguments:
        epochs: Number of training epochs.
        dataset: A tuple that contains (X_train, X_test, y_train, y_test)
        model_info: Data required to build the model.
        run_name: The name of the current run. Will be used for checkpointing.
    """

    # Create dataloaders.
    X_train, X_test, y_train, y_test = dataset
    train_ds = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_dl = DataLoader(
        train_ds,
        batch_size=training_info["batch_size"],
        shuffle=training_info["shuffle"],
        num_workers=training_info["num_workers"],
    )
    test_ds = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_dl = DataLoader(
        test_ds,
        batch_size=training_info["batch_size"],
        num_workers=training_info["num_workers"],
    )

    # Create model.
    model = SimpleModel(model_info[0], model_info[1])

    # Initialise trainer.
    trainer = Trainer(max_epochs=training_info["epochs"], log_every_n_steps=10)

    # Train model.
    trainer.fit(model, train_dl, test_dl)


if __name__ == "__main__":
    # Prepare data and run the NN.
    training_info, model_info, dataset, run_name = prepare_data()
    run(training_info, model_info, dataset, run_name)