import os
import neat
import time
import wandb
import random
import logging
import multiprocessing

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, f1_score, accuracy_score

from inatc.algos import VanillaNEAT, ECOCNEAT
from inatc.utils.helpers import read_fake_data, read_yaml, parse_arguments


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

    os.mkdir(run_name + "run_checkpoints")
    os.mkdir(run_name + "run_images")

    # Create fake dataset.
    cfg = read_yaml(data_file)
    split_size = cfg["info"]["split_size"]
    X_train, X_test, y_train, y_test = read_fake_data(
        split_size, args.seed, **cfg["dataset"]
    )

    # Initialising wandb.
    wandb.init(
        **cfg["wandb"],
        name=f"{args.run_name}_{args.seed}",
        config={
            **cfg["info"],
            **cfg["dataset"],
            "generations": cfg["generations"],
            "seed": args.seed,
        },
    )

    # Set seed.
    random.seed(args.seed)

    # Number of generations.
    n_generations = cfg["generations"]

    return X_train, X_test, y_train, y_test, run_name, n_generations, cfg["info"]["fitness_function"]


# Most of the code has been taken from the XOR example in their documentation.
# Source: https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/evolve-feedforward.py
# Config file source: https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/config-feedforward
# Code for multiprocessing referred from: https://github.com/CodeReclaimers/neat-python/blob/master/examples/openai-lander/evolve.py


def run(data, config_file):

    X_train, X_test, y_train, y_test, run_name, n_generations, fitness_evaluator = data

    # neat_model = VanillaNEAT(config_file, fitness_evaluator, run_name)
    neat_model = ECOCNEAT(config_file, fitness_evaluator, run_name)

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

    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

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
            "train_f1_macro": f1_score(y_train, train_preds, average="macro"),
            "train_f1_weighted": f1_score(y_train, train_preds, average="weighted"),
            "test_acc": accuracy_score(y_test, test_preds),
            "test_f1_macro": f1_score(y_test, test_preds, average="macro"),
            "test_f1_weighted": f1_score(y_test, test_preds, average="weighted"),
        }
    )


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    data = prepare_data()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "configs/config-feedforward")
    run(data, config_path)
