import os
import neat
import visualise
from sklearn.metrics import classification_report
from inatc.toy_experiments.utils import read_fake_data, read_yaml

# Create fake dataset.
cfg = read_yaml("run_config/dataset.yaml")
split_size, random_state = cfg["info"].values()
X_train, X_test, y_train, y_test = read_fake_data(
    split_size, random_state, **cfg["generate"]
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Most of the code has been taken from the XOR example in their documentation.
# Source: https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/evolve-feedforward.py
# Config file source: https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/config-feedforward


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(X_train, y_train):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo) ** 2


def run(config_file):
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
    p.add_reporter(neat.Checkpointer(100, filename_prefix="run_checkpoints/checkpoint"))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print("\nBest genome:\n{!s}".format(winner))

    # Evaluate on training data.
    print("*" * 50)
    print("Training Data")
    train_preds = []
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi in X_train:
        pred = winner_net.activate(xi)
        train_preds.append(1 if pred[0] > 0.5 else 0)
    print(classification_report(y_train, train_preds))
    print("*" * 50)

    # Evaluate on training data.
    print("Testing Data")
    test_preds = []
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi in X_test:
        pred = winner_net.activate(xi)
        test_preds.append(1 if pred[0] > 0.5 else 0)
    print(classification_report(y_test, test_preds))
    print("*" * 50)

    visualise.draw_net(
        config,
        winner,
        "run_images/",
        True,
        filename="toy-run",
    )
    visualise.plot_stats(stats, "run_images/", ylog=False, view=True)
    visualise.plot_species(stats, "run_images/", view=True)


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "run_config/config-feedforward")
    run(config_path)
