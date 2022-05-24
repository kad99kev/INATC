import os
import neat
import time
import random
import logging
import visualise
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from inatc.toy_experiments.utils import read_fake_data, read_yaml, parse_arguments


def prepare_data():
    """
    Parse arguments and prepare experiment data folders.
    """
    args = parse_arguments()

    global X_train, X_test, y_train, y_test, run_name, n_generations

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

    # Number of generations.
    n_generations = cfg["generations"]


# Most of the code has been taken from the XOR example in their documentation.
# Source: https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/evolve-feedforward.py
# Config file source: https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/config-feedforward


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        outputs = []
        for xi, xo in zip(X_train, y_train):
            output = net.activate(xi)
            outputs.append(np.argmax(output))
        genome.fitness = f1_score(y_train, outputs, average="weighted")


def run(config_file):
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

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(100, filename_prefix=run_name + "run_checkpoints/checkpoint")
    )

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, n_generations)

    # Finish training.
    logging.info(f"Training done! Time taken - {time.time() - start_time:.3f} seconds")

    # Display the winning genome.
    print("\nBest genome:\n{!s}".format(winner))

    # Start evaluation.
    logging.info("Starting evaluation.")
    start_time = time.time()

    # Evaluate on training data.
    logging.info("Running training evaluation...")
    train_preds = []
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi in X_train:
        pred = winner_net.activate(xi)
        train_preds.append(1 if pred[0] > 0.5 else 0)
    report = classification_report(y_train, train_preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(run_name + "train_results.csv")

    # Evaluate on testing data.
    logging.info("Running testing evaluation...")
    test_preds = []
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi in X_test:
        pred = winner_net.activate(xi)
        test_preds.append(1 if pred[0] > 0.5 else 0)
    report = classification_report(y_test, test_preds, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(run_name + "test_results.csv")

    # Finish evaluation.
    logging.info(
        f"Evaluation done! Time taken - {time.time() - start_time:.3f} seconds"
    )

    visualise.draw_net(
        config,
        winner,
        run_name + "run_images/",
        filename="toy-run",
    )
    visualise.plot_stats(stats, run_name + "run_images/", ylog=False)
    visualise.plot_species(stats, run_name + "run_images/")


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    prepare_data()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "configs/config-feedforward")
    run(config_path)
