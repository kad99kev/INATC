import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class SimpleModel(pl.LightningModule):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs - 1),
            nn.Sigmoid(),
        )
        self.loss_fn = nn.BCELoss()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, inputs):
        x = self.block(inputs)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1))
        self.train_acc(y_hat.reshape(-1), y.type("torch.IntTensor"))
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y.view(-1, 1))
        self.valid_acc(y_hat.reshape(-1), y.type("torch.IntTensor"))
        self.log("valid_acc", self.valid_acc, prog_bar=True, on_epoch=True)
        return val_loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred
