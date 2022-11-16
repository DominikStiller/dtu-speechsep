import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim

from speechsep.cli import Args
from speechsep.model import Demucs
from speechsep.util import center_trim


class LitDemucs(pl.LightningModule):
    def __init__(self, args: Args):
        super().__init__()
        self.model = Demucs(args)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y = center_trim(y, target=y_pred)
        loss = F.l1_loss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y = center_trim(y, target=y_pred)
        loss = F.l1_loss(y_pred, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
