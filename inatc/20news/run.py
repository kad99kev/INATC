import os
import neat
import time
import wandb
import random
import logging
import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.datasets import fetch_20newsgroups
from inatc.utils import read_yaml, parse_arguments, is_internet


def prepare_data():
    """
    Parse arguments and prepare experiment data folders.
    """
    args = parse_arguments()

    global X_train, X_test, y_train, y_test, target_names, run_name, n_generations

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
    X_train = np.load("data/train.npy")

    test_data = fetch_20newsgroups(subset="test", shuffle=False)
    y_test = test_data.target
    X_test = np.load("data/test.npy")

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
        settings=wandb.Settings(start_method="fork")
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


def compute_fitness(net, X_train, y_train):
    outputs = []
    for xi in X_train:
        output = neat.math_util.softmax(net.activate(xi))
        outputs.append(np.argmax(output))
    return f1_score(y_train, outputs, average="macro")


def eval_genomes(genomes, config):
    num_workers = multiprocessing.cpu_count()

    nets = []
    fitnesses = []

    for genome_id, g in genomes:
        nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))

    if num_workers < 2:
        for genome, net in tqdm(nets):
            genome.fitness = compute_fitness(net, X_train, y_train)
            fitnesses.append(genome.fitness)
    else:
        with multiprocessing.Pool(num_workers) as pool:
            jobs = []
            for genome, net in nets:
                jobs.append(pool.apply_async(compute_fitness, (net, X_train, y_train)))

            for job, (genome_id, genome) in tqdm(zip(jobs, genomes), total=len(jobs)):
                fitness = job.get(timeout=None)
                genome.fitness = fitness
                fitnesses.append(genome.fitness)
    wandb.log({"average_fitness": np.mean(fitnesses)})


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
    wandb.log({"training_time": time.time() - start_time})

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
        pred = neat.math_util.softmax(winner_net.activate(xi))
        train_preds.append(np.argmax(pred))
    report = classification_report(
        y_train, train_preds, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).transpose()
    df.to_csv(run_name + "train_results.csv")

    # Evaluate on testing data.
    logging.info("Running testing evaluation...")
    test_preds = []
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi in X_test:
        pred = neat.math_util.softmax(winner_net.activate(xi))
        test_preds.append(np.argmax(pred))
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
    prepare_data()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "configs/config-feedforward")
    run(config_path)