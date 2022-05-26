import os
import random
import time
import logging
import wandb
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

import torch
from torch.utils.data import TensorDataset, DataLoader

from model import SimpleModel
from inatc.toy_experiments.utils import read_fake_data, read_yaml, parse_arguments


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

    # Create fake dataset.
    cfg = read_yaml(data_file)
    split_size, random_state = cfg["info"].values()
    X_train, X_test, y_train, y_test = read_fake_data(
        split_size, random_state, **cfg["dataset"]
    )

    # Initialising wandb.
    # Create wandb logger.
    wandb_logger = WandbLogger(
        **cfg["wandb"],
        name=args.run_name,
        config={**cfg["info"], **cfg["dataset"], **cfg["training"]},
    )

    # Set seed.
    pl.seed_everything(random_state)
    random.seed(random_state)

    return (
        cfg["training"],
        (cfg["dataset"]["n_features"], cfg["dataset"]["n_classes"]),
        (X_train, X_test, y_train, y_test),
        run_name,
        wandb_logger,
    )


def run(training_info, model_info, dataset, run_name, wandb_logger):
    """
    Train the Neural Network.

    Arguments:
        epochs: Number of training epochs.
        dataset: A tuple that contains (X_train, X_test, y_train, y_test)
        model_info: Data required to build the model.
        run_name: The name of the current run. Will be used for checkpointing.
    """

    # Create logger.
    start_time = time.time()
    logging.basicConfig(
        filename=run_name + "run.log",
        filemode="a",
        format="%(asctime)s.%(msecs)d: %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    logging.info("Starting training.")

    # Create dataloaders.
    X_train, X_test, y_train, y_test = dataset
    train_ds = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    train_dl = DataLoader(
        train_ds,
        batch_size=training_info["batch_size"],
        shuffle=training_info["shuffle"],
        num_workers=training_info["num_workers"],
    )
    valid_ds = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
    valid_dl = DataLoader(
        valid_ds,
        batch_size=training_info["batch_size"],
        num_workers=training_info["num_workers"],
    )

    # Create model.
    model = SimpleModel(model_info[0], model_info[1])

    # Initialise trainer.
    trainer = Trainer(
        max_epochs=training_info["epochs"],
        default_root_dir=run_name,
        log_every_n_steps=len(X_train) / training_info["batch_size"],
        check_val_every_n_epoch=training_info["valid_epochs"],
        logger=wandb_logger,
    )

    # Train model.
    trainer.fit(model, train_dl, valid_dl)

    # Finish training.
    logging.info(f"Training done! Time taken - {time.time() - start_time:.3f} seconds")
    wandb.log({"training_time": time.time() - start_time})

    # Start evaluation.
    logging.info("Starting evaluation.")
    start_time = time.time()

    # Evaluate on training data.
    logging.info("Running training evaluation...")
    train_dl = DataLoader(
        train_ds,
        batch_size=len(X_train),
        num_workers=training_info["num_workers"],
    )
    train_preds = trainer.predict(dataloaders=train_dl, ckpt_path="best")
    _, train_preds = torch.max(train_preds[0].data, 1)
    report = classification_report(y_train, train_preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(run_name + "train_results.csv")

    # Evaluate on testing data.
    logging.info("Running testing evaluation...")
    test_dl = DataLoader(
        valid_ds,
        batch_size=len(X_test),
        num_workers=training_info["num_workers"],
    )
    test_preds = trainer.predict(dataloaders=test_dl, ckpt_path="best")
    _, test_preds = torch.max(test_preds[0].data, 1)
    report = classification_report(y_test, test_preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(run_name + "test_results.csv")

    # Finish evaluation.
    logging.info(
        f"Evaluation done! Time taken - {time.time() - start_time:.3f} seconds"
    )
    wandb.log({"evaluation_time": time.time() - start_time})

    # Wandb logging.
    wandb.log(
        {
            "train_acc": accuracy_score(y_train, train_preds),
            "train_f1_macro": f1_score(y_train, train_preds, average="macro"),
            "train_f1_weighted": f1_score(y_train, train_preds, average="weighted"),
            "test_acc": accuracy_score(y_test, test_preds),
            "test_f1_macro": f1_score(y_test, test_preds, average="macro"),
            "test_f1_weighted": f1_score(y_test, test_preds, average="weighted"),
        }
    )


if __name__ == "__main__":
    # Prepare data and run the NN.
    run(*prepare_data())