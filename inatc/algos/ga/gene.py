import numpy as np

import torch.nn as nn


class Gene(nn.Module):
    def __init__(
        self,
        layer_type,
        num_input_features,
        num_channels,
        num_output_features,
        kernel_size,
        layer_set,
    ):
        """
        The Gene class.
        Each gene will be a layer in the network.

        Arguments:
            layer_type: Type of layer.
            num_input_features: Number of features expected from input.
            num_channels: Number of channels expected from input.
            num_output_features: Number of features given from output.
            kernel_size: Kernel size.

        Attributes:
            layer: PyTorch equivalent of layer_type.


        """
        super().__init__()
        self.layer_type = layer_type
        self.num_input_features = num_input_features
        self.num_channels = num_channels
        self.num_output_features = num_output_features
        self.kernel_size = kernel_size
        self._build_layer(layer_set)

    def _build_layer(self, layer_set):
        """
        Build current gene as a PyTorch layer.

        Arguments:
            layer_set: Set of possible layers as key value pairs.
        """
        if self.layer_type == "linear":
            self.layer = layer_set[self.layer_type](
                self.num_input_features, self.num_output_features
            )
        elif self.layer_type in ["lstm", "gru"]:
            self.layer = layer_set[self.layer_type](
                self.num_input_features, self.num_output_features, batch_first=True
            )
        elif self.layer_type == "conv":
            self.layer = layer_set[self.layer_type](
                self.num_channels, self.num_output_features, self.kernel_size
            )
        elif "pool" in self.layer_type:
            self.layer = layer_set[self.layer_type](self.kernel_size)
        elif "norm" in self.layer_type:
            self.layer = layer_set[self.layer_type](self.num_channels)

    def calculate_output_shape(self, final_layer=False):
        """
        Calculate output shape of current layer to use as inputs for next layer.

        Arguments:
            final_layer: If current layer is the final layer, flatten the output shape.
        """
        output_shape = None
        if self.layer_type in ["linear", "lstm", "gru"]:
            output_shape = (self.num_channels, self.num_output_features)
        if self.layer_type == "conv":
            feature_shape = (
                self.num_input_features
                + 2 * self.layer.padding[0]
                - self.layer.dilation[0] * (self.kernel_size - 1)
                - 1
            ) / self.layer.stride[0] + 1
            output_shape = (self.num_output_features, np.floor(feature_shape))

        if self.layer_type == "max_pool":
            feature_shape = (
                self.num_input_features
                + 2 * self.layer.padding
                - self.layer.dilation * (self.kernel_size - 1)
                - 1
            ) / self.layer.stride + 1
            output_shape = (self.num_channels, np.floor(feature_shape))

        if self.layer_type == "avg_pool":
            feature_shape = (
                self.num_input_features + 2 * self.layer.padding[0] - self.kernel_size
            ) / self.layer.stride[0] + 1
            output_shape = (self.num_channels, np.floor(feature_shape))
        if "norm" in self.layer_type:
            output_shape = (self.num_channels, self.num_input_features)

        output_shape = tuple(map(lambda x: int(x), output_shape))
        return np.prod(output_shape) if final_layer else output_shape

    def forward(self, x):
        """
        Forward pass for the layer.

        Arguments:
            x: Input vector.
        """
        if self.layer_type in ["lstm", "gru"]:
            out, _ = self.layer(x)
            return out
        return self.layer(x)
