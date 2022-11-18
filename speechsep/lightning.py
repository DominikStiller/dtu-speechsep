import pytorch_lightning as pl
import torch.nn.functional as F
from pesq import NoUtterancesError
from torch import optim
from torchmetrics import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from speechsep.cli import Args
from speechsep.model import Demucs
from speechsep.util import center_trim


class LitDemucs(pl.LightningModule):
    def __init__(self, args: Args):
        super().__init__()
        self.model = Demucs(args)

        self.metric_sisdr = ScaleInvariantSignalDistortionRatio()
        self.metric_pesq = PerceptualEvaluationSpeechQuality(8000, "nb")
        self.metric_stoi = ShortTimeObjectiveIntelligibility(8000, False)

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
        loss, sisdr, pesq, stoi = self._shared_eval_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log_dict({"val_sisdr": sisdr, "val_pesq": pesq, "val_stoi": stoi}, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, sisdr, pesq, stoi = self._shared_eval_step(batch)
        self.log("test_loss", loss, sync_dist=True)
        self.log_dict({"test_sisdr": sisdr, "test_pesq": pesq, "test_stoi": stoi}, sync_dist=True)
        return loss

    def _shared_eval_step(self, batch):
        x, y = batch
        y_pred = self(x)
        y = center_trim(y, target=y_pred)
        loss = F.l1_loss(y_pred, y)

        # Calculate average metrics over all channels and examples
        sisdr = self.metric_sisdr(y_pred, y)
        try:
            pesq = self.metric_pesq(y_pred, y)
        except NoUtterancesError:
            pesq = 0.0
        stoi = self.metric_stoi(y_pred, y)

        return loss, sisdr, pesq, stoi

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
