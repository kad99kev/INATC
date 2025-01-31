import os
import neat
import time
import wandb
import random
import logging
import multiprocessing

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, classification_report, accuracy_score, log_loss

from inatc.algos import VanillaNEAT, ECOCNEAT
from inatc.utils.helpers import read_yaml, parse_arguments, is_internet


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

    # Get config file based on type of experiment.
    data_file = f"configs/{args.config}"
    cfg = read_yaml(data_file)

    os.mkdir(run_name + "run_checkpoints")
    os.mkdir(run_name + "run_images")

    # Load dataset
    train_data = fetch_20newsgroups(subset="train", shuffle=False)
    y_train, target_names = train_data.target, train_data.target_names
    X_train = np.load(args.train)

    test_data = fetch_20newsgroups(subset="test", shuffle=False)
    y_test = test_data.target
    X_test = np.load(args.test)

    # Initialising wandb.
    wandb.init(
        **cfg["wandb"],
        name=f"{args.run_name}_{args.seed}",
        config={
            **cfg["info"],
            "generations": cfg["generations"],
            "seed": args.seed,
        },
        mode="online" if is_internet() else "offline",
        settings=wandb.Settings(start_method="fork"),
    )

    # Set seed.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Shuffle data.
    shuff_idxs = np.random.permutation(len(X_train))
    X_train, y_train = X_train[shuff_idxs], y_train[shuff_idxs]

    shuff_idxs = np.random.permutation(len(X_test))
    X_test, y_test = X_test[shuff_idxs], y_test[shuff_idxs]

    # Number of generations.
    n_generations = cfg["generations"]

    # Print number of available CPUs.
    print(f"Number of available CPUs: {multiprocessing.cpu_count()}")

    return (
        (
            X_train,
            X_test,
            y_train,
            y_test,
            target_names,
            run_name,
            n_generations,
            cfg["info"]["fitness_function"],
            cfg["info"]["model"],
        ),
        args.neat_config,
    )


def run(data, config_file):

    (
        X_train,
        X_test,
        y_train,
        y_test,
        target_names,
        run_name,
        n_generations,
        fitness_evaluator,
        model,
    ) = data

    if model == "vanilla":
        neat_model = VanillaNEAT(config_file, fitness_evaluator, run_name)
    elif model == "ecoc":
        neat_model = ECOCNEAT(config_file, fitness_evaluator, run_name)
    else:
        raise Exception(
            f"{model} does not exist! Please choose between `vanilla` or `ecoc`."
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

    neat_model.train(X_train, y_train, n_generations)

    # Finish training.
    logging.info(f"Training done! Time taken - {time.time() - start_time:.3f} seconds")
    wandb.log({"training_time": time.time() - start_time})

    # Start evaluation.
    logging.info("Starting evaluation.")
    start_time = time.time()

    # Evaluate on training data.
    logging.info("Running training evaluation...")
    train_preds = neat_model.predict(X_train)
    report = classification_report(
        y_train, train_preds, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).transpose()
    df.to_csv(run_name + "train_results.csv")

    # Evaluate on testing data.
    logging.info("Running testing evaluation...")
    test_preds = neat_model.predict(X_test)
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

    # visualise.draw_net(
    #     config,
    #     winner,
    #     run_name + "run_images/",
    #     filename="toy-run",
    # )
    # visualise.plot_stats(stats, run_name + "run_images/", ylog=False)
    # visualise.plot_species(stats, run_name + "run_images/")

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
    data, neat_config = prepare_data()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, f"configs/{neat_config}")
    run(data, config_path)
