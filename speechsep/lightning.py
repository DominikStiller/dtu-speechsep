import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim

from speechsep.cli import Args
from speechsep.model import Demucs
from speechsep.util import center_trim

from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


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

    def validation_step(self, batch, batch_idx):
        loss, pesq, stoi = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss, "val_pesq": pesq, "val_stoi": stoi}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, pesq, stoi = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_loss": loss, "test_pesq": pesq, "test_stoi": stoi}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y = center_trim(y, target=y_pred)
        loss = F.l1_loss(y_pred, y)
        # calculate pesq and stoi
        nb_pesq = PerceptualEvaluationSpeechQuality(8000, 'nb')
        pesq = nb_pesq(y_pred.sum(1), y.sum(1))
        # pesq0 = nb_pesq(y_pred[:, 0:1, :], y[:, 0:1, :])
        # pesq1 = nb_pesq(y_pred[:, 1:2, :], y[:, 1:2, :])
        stoi = ShortTimeObjectiveIntelligibility(8000, False)
        stoi = stoi(y_pred.sum(1), y.sum(1))
        return loss, pesq, stoi

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
