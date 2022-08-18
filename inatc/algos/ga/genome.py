import random
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm

from .gene import Gene


class Genome(pl.LightningModule):
    def __init__(self, config, multi_class=True):
        """
        The Genome class.
        Each genome is a neural network with multiple layers.
        The geneome has a list of genes wherein each gene represents a layer in the network.

        Arguments:
            config: A dictionary containing configuration information.
            multi_class: If the genome is being built for a multi-class task.

        Attributes:
            num_channels: Number of input channels from the language model.
            num_features: Number of input features from the language model.
            num_classes: Number of output classes for the current task.
            layer_set: Key value pairs of layer names and their PyTorch equivalent.
            layer_config: Key value pairs of layer configuration.
            mutation_config: Key value pairs of mutation configuration.
            fitness: Current fitness of the genome.
            loss_fn: The loss function to be used while training. Depends on the type of task (whether multi-class or multi-label).
            layers: List of PyTorch layers determined by genes.
            genes: List of genes (of class `Gene`).
            genes_list: List of genes in tuple form.
        """

        super().__init__()
        # Evolution setup.
        self.num_channels = config["num_input_channels"]
        self.num_features = config["num_input_features"]
        self.num_classes = config["num_output_classes"]
        self.layer_set = config["layer_set"]
        self.activation_set = config["activation_set"]
        self.layer_config = config["layer_config"]
        self.mutation_config = config["mutation_config"]
        self.multi_class = multi_class
        self.fitness = None

        # Training setup.
        if self.multi_class:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.BCELoss()

    def _build_network(self):
        """
        Builds the neural network with PyTorch layers using its genes.
        """
        self.layers = nn.ModuleList()

        # Only build layers if genes are present.
        if len(self.genes) > 0:
            for gene in self.genes:
                self.layers.append(gene)
            last_layer_out = gene.calculate_output_shape(final_layer=True)
        else:
            # No genes present, last layer features are obtained directly from BERT.
            last_layer_out = np.prod((self.num_channels, self.num_features))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(last_layer_out, self.num_classes))
        # If multi-label then add Sigmoid layer.
        if not self.multi_class:
            self.layers.append(nn.Sigmoid())

    def _process_genes_list(self, num_channels, num_features, genes_list):
        """
        Function to process genes and build network.

        Arguments:
            num_channels: Number of input channels from the language model.
            num_features: Number of input features from the language model.
            genes_list: List of genes for the genome.
        """
        self.genes = []
        self.genes_list = []

        for gene in genes_list:

            # Avoid layers that cannot be processed.
            if num_features <= 1 and gene[0] != "linear":
                continue

            # type, activation, num_out, kernel
            g = Gene(
                gene[0],
                gene[1],
                num_features,
                num_channels,
                gene[2],
                gene[3],
                self.layer_set,
                self.activation_set,
            )
            check_channels, check_features = g.calculate_output_shape()

            # Avoid any layers that might be invalid.
            if check_channels < 1 or check_features < 1:
                continue

            # Only change values if valid.
            num_channels, num_features = check_channels, check_features
            self.genes.append(g)
            self.genes_list.append(gene)

        self._build_network()

    def _label_transform(self, preds):
        """
        Convert predictions into label format depending on task type.

        Arguments:
            preds: Prediction vector.
        """
        if self.multi_class:
            _, preds = torch.max(preds.data, 1)
            return preds.cpu().numpy()
        else:
            preds = preds.cpu().numpy()
            preds[preds > 0.5] = 1
            preds[preds != 1] = 0
            return preds

    def _mutate_layer(self):
        """
        Mutate layer type.
        """
        return random.choice(list(self.layer_set.keys()))

    def _mutate_output_features(self, org_value):
        """
        Mutate the number of output features.

        Arguments:
            org_value: Current value of the output features.
        """
        new_val = org_value + round(
            random.gauss(0.0, self.mutation_config["mutate_power"]) * 100
        )
        min_value, max_value = self.layer_config["output_feature_range"]
        return max(min(new_val, max_value), min_value)

    def _mutate_kernel_size(self, org_value):
        """
        Mutate the kernel size.

        Arguments:
            org_value: Current value of kernel size.
        """
        new_val = org_value + round(
            random.gauss(0.0, self.mutation_config["mutate_power"]) * 10
        )
        min_value, max_value = self.layer_config["kernel_size_range"]
        return max(min(new_val, max_value), min_value)

    def _mutate_delete_layer(self):
        """
        Deletes a layer.
        """
        idx = random.randint(0, len(self.genes_list) - 1)
        return idx

    def _mutate_add_layer(self):
        """
        Creates a new layer.
        """
        layer_type = random.choice(list(self.layer_set.keys()))
        layer_num_outputs = random.randint(*self.layer_config["output_feature_range"])
        layer_kernel_size = random.randint(*self.layer_config["kernel_size_range"])
        idx = random.randint(0, len(self.genes_list))
        return (idx, [layer_type, layer_num_outputs, layer_kernel_size])

    def create_genes(self):
        """
        Creates genes for the genome. Called when initially populating the population.
        """
        max_num = random.randint(*self.layer_config["num_layers_range"])
        genes_list = []
        for i in range(max_num):
            layer_type = random.choice(list(self.layer_set.keys()))
            activation_type = random.choice(list(self.activation_set.keys()))
            layer_num_outputs = random.randint(
                *self.layer_config["output_feature_range"]
            )
            layer_kernel_size = random.randint(*self.layer_config["kernel_size_range"])
            genes_list.append(
                [layer_type, activation_type, layer_num_outputs, layer_kernel_size]
            )

        self._process_genes_list(self.num_channels, self.num_features, genes_list)

    def perform_crossover(self, parent_1, parent_2):
        """
        Perform crossover between two parents.

        Arguments:
            parent_1: First crossover parent.
            parent_2: Second crossover parent.
        """
        # Sort parents based on their fitness.
        if parent_1.fitness > parent_2.fitness:
            parent_1, parent_2 = parent_1, parent_2
        else:
            parent_1, parent_2 = parent_2, parent_1

        # Perform crossover.
        p_1_genes, p_2_genes = parent_1.genes_list, parent_2.genes_list
        genes_list = []
        for i in range(len(p_1_genes)):

            # Randomly choose which genes are selected in crossover.
            if i < len(p_2_genes):
                if random.random() > 0.5:
                    genes_list.append(p_1_genes[i])
                else:
                    genes_list.append(p_2_genes[i])
            else:
                # If parent_1 has more layers then copy the layers since it is fitter.
                genes_list.append(p_1_genes[i])

        self._process_genes_list(self.num_channels, self.num_features, genes_list)

    def mutate(self):
        """
        Perform mutation.
        """
        genes_list = self.genes_list.copy()

        # Only delete if there is something to delete.
        if (
            len(genes_list) > 0
            and random.random() < self.mutation_config["mutate_delete_layer_prob"]
        ):
            idx = self._mutate_delete_layer()
            del genes_list[idx]

        if random.random() < self.mutation_config["mutate_add_layer_prob"]:
            new_gene = self._mutate_add_layer()
            genes_list.insert(new_gene[0], new_gene[1])

        for i in range(len(genes_list)):
            if random.random() < self.mutation_config["mutate_layer_prob"]:
                genes_list[i][0] = self._mutate_layer()
            if random.random() < self.mutation_config["mutate_num_output_prob"]:
                genes_list[i][1] = self._mutate_output_features(genes_list[i][1])
            if random.random() < self.mutation_config["mutate_num_kernel_prob"]:
                genes_list[i][2] = self._mutate_kernel_size(genes_list[i][2])

        self._process_genes_list(self.num_channels, self.num_features, genes_list)

    # The functions below belong to the PyTorch Lightning syntax.
    # For more information read: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        pred = self(batch[0])
        pred = self._label_transform(pred)
        return pred

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = self.layers[i](x)
        return x
