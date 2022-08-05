import multiprocessing
import os
import gzip
import wandb
import random
import pickle
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPSpawnStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from .genome import Genome

import warnings

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

class Population:
    def __init__(self, config, seed, accelerator=None, devices=None, save_path=".", multi_class=False):
        """
        The Population class.
        A population will have multiple Genomes as individuals.

        Arguments:
            config: A dictionary containing configuration information.
            seed: Random seed to be used.
            accelerator: Device accelerator for PyTorch Lightning.
            devices: Number of accelerator devices for PyTorch Lightning.
            save_path: Base path for creating checkpoints.
            multi_class: If the genome is being built for a multi-class task.

        Attributes:
            training_config: Configuration information for training.
            evolution_config: Configuration information for evolution.
            best_genome: Best genome observed throughout training.
            base_path: Base path for checkpoint saving.
            fitness_evaluator: Fitness function to be used.
        """
        self.training_config = config["training"]
        self.evolution_config = config["evolution"]
        self.fitness_evaluator = self.training_config["fitness_function"]
        self.accelerator = accelerator
        self.devices = devices
        self.multi_class = multi_class
        self.base_path = save_path
        self.best_genome = None
        self.seed = seed

        # Set seed.
        random.seed(seed)
        pl.seed_everything(seed)

    def _compute_fitness(self, y, preds):
        """
        Compute fitness given target and predictions.

        Arguments:
            y: Truth labels.
            preds: Prediction labels.
        """
        if self.fitness_evaluator == "f1_score":
            return f1_score(y, preds, average="macro", zero_division=0)
        if self.fitness_evaluator == "accuracy_score":
            return accuracy_score(y, preds)
        if self.fitness_evaluator == "log_loss":
            return 1 - log_loss(y, preds)

    def _populate(self):
        """
        Used to create the initial population.
        """
        population = []
        for _ in tqdm(range(self.training_config["population_size"])):
            genome = Genome(self.evolution_config, multi_class=self.multi_class)
            genome.create_genes()
            population.append(genome)
        self.population = population

    def _create_checkpoint_dir(self, path):
        """
        Create a checkpoint directory if it does not exist.

        Arguments:
            path: Path at which checkpoint directory would be created.
        """
        path = f"{self.base_path}{path}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return gzip.open(path, "w", compresslevel=5)

    def _create_checkpoint(self, generation):
        """
        Create checkpoint.

        Arguments:
            generation: The current generation for which checkpoints are being saved.
        """
        genome_genes = []
        for genome in self.population:
            genome_genes.append(genome.genes_list)
        data = (
            genome_genes,
            random.getstate(),
            generation,
            self.training_config,
            self.evolution_config,
        )
        with self._create_checkpoint_dir(f"genes/{generation}/genomes.pickle") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _prepare_dataloader(self, X, y=None, shuffle=False, predict=False):
        """
        Prepare a PyTorch DataLoader.

        Arguments:
            X: The input features.
            y: The output targets.
            shuffle: If DataLoader should be shuffled.
            predict: If building DataLoader for prediction.
        """
        if predict:
            dataset = TensorDataset(torch.Tensor(X))
        else:
            dataset = TensorDataset(
                torch.Tensor(X),
                torch.LongTensor(y) if self.multi_class else torch.FloatTensor(y),
            )

        # Dataloader info.
        pin_memory = False
        if self.accelerator == "gpu":
            pin_memory = True

        return DataLoader(
            dataset,
            self.training_config["batch_size"],
            shuffle=shuffle,
            num_workers=self.training_config["num_workers"],
            persistent_workers=True,
            pin_memory=pin_memory,
        )

    def _save_best_genome(self, best_genome, best_trainer):
        """
        Save best genome seen during training.

        Arguments:
            best_genome: The best genome to be saved.
            best_trainer: To save the PyTorch Lightning model checkpoint.
        """
        data = best_genome.genes_list
        best_trainer.save_checkpoint(
            f"{self.base_path}checkpoints/best_genome/best_genome.ckpt"
        )
        with self._create_checkpoint_dir("genes/best_genome/best_genome.pickle") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, path):
        """
        Load checkpoint given a path.

        Arguments:
            path: Path of checkpoint to be loaded.
        """
        with gzip.open(path) as f:
            data = pickle.load(f)
        return data

    def run(self, train_data, test_data):
        """
        Start training.

        Arguments:
            train_data: Training data.
            test_data: Testing data.
        """

        # Prepare data.
        X_train, y_train = train_data
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=self.training_config["validation_split"],
            random_state=self.seed,
        )
        X_test, y_test = test_data

        print(f"Training data size - {len(X_train)}")
        print(f"Validation data size - {len(X_valid)}")
        print(f"Testing data size - {len(X_test)}")

        # Training DataLoader.
        train_dl = self._prepare_dataloader(
            X_train, y_train, shuffle=self.training_config["shuffle"]
        )

        # Validation DataLoader.
        valid_dl = self._prepare_dataloader(X_valid, y_valid)

        # Get strategy based on available machines.
        strategy = None
        if accelerator == "gpu":
            # No other DDP strategy works for PyTorch Lightning when using GPU.
            strategy = DDPSpawnStrategy(find_unused_parameters=False)

        # Populate initial population.
        self._populate()

        for gen in range(1, self.training_config["num_generations"] + 1):

            print(
                f"{'*' * 10} Starting Generation {gen} / {self.training_config['num_generations']} {'*' * 10}"
            )

            # To track average fitness for each generation.
            fitnesses = []
            for i, genome in enumerate(self.population):

                # Create checkpoint for models.
                checkpointing = False
                callbacks = []

                if gen % self.training_config["checkpoint_interval"] == 0:
                    checkpointing = True
                    checkpoint_callback = ModelCheckpoint(
                        dirpath=f"{self.base_path}checkpoints/{gen}/",
                        filename=f"genome{i}",
                        save_weights_only=True,
                    )
                    callbacks.append(checkpoint_callback)

                earlystopping_callback = EarlyStopping(
                    monitor="val_loss", min_delta=0.00, patience=3, mode="min"
                )
                callbacks.append(earlystopping_callback)

                # Initialise trainer.
                trainer = Trainer(
                    default_root_dir=f"{self.base_path}logs/{gen}/",
                    max_epochs=self.training_config["epochs"],
                    callbacks=callbacks,
                    enable_checkpointing=checkpointing,
                    accelerator=self.accelerator,
                    strategy=strategy,
                    devices=self.devices,
                )
                # Train model.
                trainer.fit(genome, train_dl, valid_dl)

                # Perform evaluation.
                print("Performing evaluation...")
                test_preds = self.predict(X_test, genome)
                score = self._compute_fitness(y_test, test_preds)
                genome.fitness = score
                print("Fitness: ", genome.fitness)
                fitnesses.append(genome.fitness)

                if (
                    self.best_genome is None
                    or genome.fitness > self.best_genome.fitness
                ):
                    print(f"Updating Best Genome with Fitness: {genome.fitness}")
                    self.best_genome = genome
                    self._save_best_genome(genome, trainer)

            # Log average fitness from population.
            wandb.log({"average_fitness": np.mean(fitnesses)})

            # Perform checkpointing.
            if checkpointing:
                self._create_checkpoint(gen)

            # Perform reproduction
            new_population = []
            for _ in tqdm(range(len(self.population))):
                p_1, p_2 = random.choices(
                    self.population, weights=[g.fitness for g in self.population], k=2
                )
                genome = Genome(self.evolution_config, multi_class=self.multi_class)
                genome.perform_crossover(p_1, p_2)
                genome.mutate()
                new_population.append(genome)
            self.population = new_population

        return self.best_genome

    def predict(self, X, genome=None):
        """
        Run predictions for given data.

        Arguments:
            X: Input data.
            genome: If provided, it will run prediction using given genome. \
                Else it will use best genome.
            trainer: If provided, it will use the given trainer. \
                Else it will create a new one.
            accelerator: Device accelerator for PyTorch Lightning.
            devices: Number of accelerator devices for PyTorch Lightning.
        """
        if genome is None and not hasattr(self, "best_genome"):
            raise RuntimeError(
                "You must train the model before prediction. Run the .run() function to begin training."
            )
        if genome is None:
            print(
                f"Predicting with best genome with fitness: {self.best_genome.fitness}"
            )
            genome = self.best_genome
        dl = self._prepare_dataloader(X, predict=True)
        
        devices = None
        if self.accelerator == "gpu" or self.accelerator == "mps":
            # Only use one device for predictions.
            devices = 1
        trainer = Trainer(accelerator=self.accelerator, devices=devices)
        preds = np.concatenate(trainer.predict(genome, dataloaders=dl), axis=0)
        return preds
