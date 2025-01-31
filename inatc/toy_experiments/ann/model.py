import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class SimpleModel(pl.LightningModule):
    """
    Simple Feedforward Neural Network with Linear Layers.

    The model has been implemented using PyTorch Lightning.
    For more information, see - https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction.html#define-a-lightningmodule

    Arguments:
        num_inputs: Number of input features.
        num_classes: Number of output classes.
    """

    def __init__(self, num_inputs, num_classes):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.F1Score(average="macro", num_classes=num_classes)
        self.valid_acc = torchmetrics.F1Score(average="macro", num_classes=num_classes)

    def forward(self, inputs):
        x = self.block(inputs)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.train_acc(y_hat, y)
        self.log(
            "train_f1", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.valid_acc(y_hat, y)
        self.log("valid_f1", self.valid_acc, prog_bar=True, on_epoch=True)
        return val_loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred
