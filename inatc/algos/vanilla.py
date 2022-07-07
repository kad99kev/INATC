import neat
import wandb
import multiprocessing

import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

class VanillaNEAT:

    def __init__(self, config_file, fitness_evaluator, run_name):
        self.config_file = config_file
        self.run_name = run_name
        self.fitness_evaluator = fitness_evaluator

    
    def _fitness_function(self, preds):
        if self.fitness_evaluator == "f1_score":
            return f1_score(self.y_train, preds, average="macro")
        if self.fitness_evaluator == "accuracy_score":
            return accuracy_score(self.y_train, preds)
    
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

        # Load configuration.
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_file,
        )

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
        self.winner_genome = p.run(self.eval_genomes, n_generations)
        self.winner_net = neat.nn.FeedForwardNetwork.create(self.winner_genome, config)


    def predict(self, X):
        if not hasattr(self, "winner_genome"):
            raise RuntimeError("You must train the model before prediction. Run the .train() function to begin training.")
        
        preds = []
        for x in X:
            pred = neat.math_util.softmax(self.winner_net.activate(x))
            preds.append(np.argmax(pred))
        return preds