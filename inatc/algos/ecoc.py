import neat
import time
import wandb
import logging
import multiprocessing

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import accuracy_score, euclidean_distances, f1_score

class ECOCNEAT:

    def __init__(self, config_file, fitness_evaluator, run_name, code_size=1.5):
        self.config_file = config_file
        self.run_name = run_name
        self.fitness_evaluator = fitness_evaluator
        self.code_size = code_size

    
    def _fitness_function(self, preds):
        if self.fitness_evaluator == "f1_score":
            return f1_score(self.y_, preds, average="macro")
        if self.fitness_evaluator == "accuracy_score":
            return accuracy_score(self.y_, preds)
    
    def compute_fitness(self, net):
        try:
            outputs = []
            for xi in self.X_train:
                output = neat.math_util.softmax(net.activate(xi))
                outputs.append(np.argmax(output))
            return self._fitness_function(outputs)
        except OverflowError:
            return 0
    
    def eval_genomes(self, genomes, config):
        num_workers = multiprocessing.cpu_count()

        nets = []
        fitnesses = []

        for genome_id, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))

        if num_workers < 2:
            for genome, net in tqdm(nets):
                genome.fitness = self.compute_fitness(net)
                fitnesses.append(genome.fitness)
        else:
            with multiprocessing.Pool(num_workers) as pool:
                jobs = []
                for genome, net in nets:
                    jobs.append(pool.apply_async(self.compute_fitness, (net, )))

                for job, (genome_id, genome) in tqdm(zip(jobs, genomes), total=len(jobs)):
                    fitness = job.get(timeout=None)
                    genome.fitness = fitness
                    fitnesses.append(genome.fitness)
        wandb.log({"average_fitness": np.mean(fitnesses)})

    def train(self, X_train, y_train, n_generations):

        self.X_train, self.y_train = X_train, y_train

        self.classes = np.unique(self.y_train)
        self.n_classes = len(self.classes)

        code_size_ = int(self.n_classes * self.code_size)
        self.code_book = np.random.uniform(size=(self.n_classes, code_size_))
        self.code_book[self.code_book > 0.5] = 1
        self.code_book[self.code_book != 1] = 0

        classes_index = {c: i for i, c in enumerate(self.classes)}

        Y = np.array([self.code_book[classes_index[y_train[i]]] for i in range(len(self.y_train))])


        # Load configuration.
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_file,
        )

        self.winner_genomes = []
        self.winner_nets = []

        for i in range(Y.shape[1]):

            print(f"\n{'*' * 10} Classifier Number: {i} {'*' * 10}")

            self.y_ = Y[:, i]

            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(config)

            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(
                neat.Checkpointer(100, filename_prefix=self.run_name + "run_checkpoints/checkpoint")
            )

            # Run for up to N generations.
            winner = p.run(self.eval_genomes, n_generations // Y.shape[1])
            winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
            self.winner_genomes.append(winner)
            self.winner_nets.append(winner_net)

    def predict(self, X):
        if not hasattr(self, "winner_genomes"):
            raise RuntimeError("You must train the model before prediction. Run the .train() function to begin training.")
        
        pred_arr = []
        for winner_net in self.winner_nets:
            net_preds = []
            for x in X:
                pred = neat.math_util.softmax(winner_net.activate(x))
                net_preds.append(pred[1])
            pred_arr.append(net_preds)
        pred_arr = np.array(pred_arr).T
        pred_dist = euclidean_distances(pred_arr, self.code_book).argmin(axis=1)
        return self.classes[pred_dist]