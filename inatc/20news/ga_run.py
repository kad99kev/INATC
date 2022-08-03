import os
import time
import wandb
import logging
import multiprocessing

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, classification_report, accuracy_score, log_loss

from configs.ga_config import GAConfig
from inatc.algos.ga import Population
from inatc.utils.helpers import parse_arguments, get_accelerator, is_internet


def prepare_data():
    """
    Parse arguments and prepare experiment data folders.
    """
    args = parse_arguments()

    # Create experiment directories based on the type of experiment.
    run_name = "runs/" + f"{args.run_name}_{args.seed}" + "/"
    if os.path.isdir(run_name):
        raise FileExistsError(
            "A previous run state already exists! Please rename the previous run or delete the folder."
        )
    else:
        os.mkdir(run_name)

    # Load dataset
    print("Reading data...")
    train_data = fetch_20newsgroups(subset="train", shuffle=False)
    y_train, target_names = train_data.target, train_data.target_names
    X_train = np.load(args.train)

    test_data = fetch_20newsgroups(subset="test", shuffle=False)
    y_test = test_data.target
    X_test = np.load(args.test)
    print("Finished reading!")

    # Print number of available CPUs.
    print(f"Number of available CPUs: {multiprocessing.cpu_count()}")

    # Print accelerator and devices.
    accelerator, devices = get_accelerator()
    print(f"Using {accelerator} accelerator with {devices} device(s)")

    return (X_train, X_test, y_train, y_test, args.seed, accelerator, devices, run_name)


def run(data, config_data):

    (
        X_train,
        X_test,
        y_train,
        y_test,
        seed,
        accelerator,
        devices,
        run_name,
    ) = data

    # Initialising wandb.
    wandb_config = config_data["wandb"]
    evolution_config = config_data["evolution"]
    wandb.init(
        **wandb_config if wandb_config is not None else wandb_config,
        config={
            **config_data["training"],
            **evolution_config["mutation_config"],
            **evolution_config["layer_config"],
            "layer_set": [layer for layer in evolution_config["layer_set"].keys()],
            "num_input_channels": evolution_config["num_input_channels"],
            "num_input_features": evolution_config["num_input_features"],
            "num_output_classes": evolution_config["num_output_classes"],
        },
        mode="online" if is_internet() else "offline",
        settings=wandb.Settings(start_method="fork"),
    )

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

    ga_model = Population(config_data, seed, save_path=run_name, multi_class=True)
    best_model = ga_model.run(
        (X_train, y_train),
        (X_test, y_test),
        accelerator=accelerator,
        devices=devices,
    )

    # Finish training.
    logging.info(f"Training done! Time taken - {time.time() - start_time:.3f} seconds")
    wandb.log({"training_time": time.time() - start_time})

    # Start evaluation.
    logging.info("Starting evaluation.")
    start_time = time.time()

    # Evaluate on training data.
    logging.info("Running training evaluation...")
    train_preds = ga_model.predict(X_train)
    report = classification_report(
        y_train, train_preds, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).transpose()
    df.to_csv(run_name + "train_results.csv")

    # Evaluate on testing data.
    logging.info("Running testing evaluation...")
    test_preds = ga_model.predict(X_test)
    report = classification_report(
        y_test, test_preds, output_dict=True, zero_division=0
    )
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
            "train_f1_micro": f1_score(y_train, train_preds, average="micro"),
            "train_f1_macro": f1_score(y_train, train_preds, average="macro"),
            "train_f1_weighted": f1_score(y_train, train_preds, average="weighted"),
            "test_acc": accuracy_score(y_test, test_preds),
            "test_f1_micro": f1_score(y_test, test_preds, average="micro"),
            "test_f1_macro": f1_score(y_test, test_preds, average="macro"),
            "test_f1_weighted": f1_score(y_test, test_preds, average="weighted"),
        }
    )


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    data = prepare_data()
    run(data, GAConfig)
