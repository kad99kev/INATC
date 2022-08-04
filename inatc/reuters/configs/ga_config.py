import torch.nn as nn

"""
Evolution:
    layer_set: Key value pairs of layer names and their PyTorch equivalent.
    layer_config:
        output_feature_range: Range of output nodes for a layer. \
            Not applicable for pooling and normalisation layers.
        num_layers_range: Range of layers each genome can have.
        kernel_size_range: Range of the kernel size for a layer. \
            Only applicable to convolution and pooling layers.
    mutation_config:
        mutate_layer_prob: Probability for a layer type to mutate.
        mutate_num_output_prob: Probabilitiy for `output_features` to mutate.
        mutate_num_kernel_prob: Probabilitiy for `kernel_size` to mutate.
        mutate_delete_layer_prob: Probabilitiy for a layer to be deleted.
        mutate_add_layer_prob: Probabilitiy for a new layer to be added.
        mutate_power: Standard deviation value for a Gaussian Distribution. \
            `mutate_power` is used when adjusting the `output_features` and \
                `kernel_size`
    num_input_channels: Number of input channels from the language model.
    num_input_features: Number of input features from the language model.
    num_output_classes: Number of output classes for the current task.
Training:
    population_size: Number of individuals in a population.
    num_generations: Number of generations to perform evolution for.
    checkpoint_interval: Number of generations after which checkpoints are saved.
    batch_size: Batch size for model training.
    epochs: Number of epochs per model per generation. (Early stopping has been implemented as well.)
    shuffle: Whether to shuffle dataloaders.
    num_workers: Number of processes to initiate to load data batch. (Mainly used by PyTorch DataLoader.)
    validation_split: Validation split for Early Stopping. It splits the training data.
    seed: Random seed for reproducibility.
    fitness_function: Fitness function for evaluation.
Wandb:
    project: Name of Weights and Biases project.
    entity: Entity name of Weights and Biases project.
"""

GAConfig = {
    "evolution": {
        "layer_set": {
            "linear": nn.Linear,
            "conv": nn.Conv1d,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "max_pool": nn.MaxPool1d,
            "avg_pool": nn.AvgPool1d,
            "batch_norm": nn.BatchNorm1d,
            "instance_norm": nn.InstanceNorm1d,
        },
        "layer_config": {
            "output_feature_range": (1, 512),
            "num_layers_range": (1, 2),
            "kernel_size_range": (1, 9),
        },
        "mutation_config": {
            "mutate_layer_prob": 0.1,
            "mutate_num_output_prob": 0.3,
            "mutate_num_kernel_prob": 0.3,
            "mutate_delete_layer_prob": 0.2,
            "mutate_add_layer_prob": 0.2,
            "mutate_power": 0.2,
        },
        "num_input_channels": 512,
        "num_input_features": 256,
        "num_output_classes": 90,
    },
    "training": {
        "population_size": 150,
        "num_generations": 10,
        "checkpoint_interval": 2,
        "batch_size": 32,
        "epochs": 10,
        "shuffle": True,
        "validation_split": 0.2,
        "seed": 0,
        "fitness_function": "f1_score",
    },
    "wandb": {"project": "inatc-toy", "entity": "kad99kev"},
}
