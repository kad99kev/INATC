import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y.view(-1, 1))
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_fn(y_hat, y.view(-1, 1))
        self.log("test_loss", test_loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred
