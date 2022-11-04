import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim

from speechsep.model import Demucs
from speechsep.plotting import plot_separated_with_truth
from speechsep.util import center_trim


class LitDemucs(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Demucs()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y = center_trim(y, target=y_pred)
        loss = F.mse_loss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y = center_trim(y, target=y_pred)
        loss = F.mse_loss(y_pred, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer


if __name__ == "__main__":
    from speechsep.mock_dataset import SinusoidDataset
    from torch.utils.data import DataLoader
    import os

    train = False

    is_hpc = "LSF_ENVDIR" in os.environ
    use_gpu = is_hpc and torch.cuda.is_available()
    test_checkpoint_path = "data/lightning_logs/version_4/checkpoints/epoch=19-step=1280.ckpt"

    if use_gpu:
        dataloader_args = {"batch_size": 16, "num_workers": 4, "persistent_workers": True}
        trainer_args = {
            "max_epochs": 20,
            "accelerator": "gpu",
            "devices": 1,
            "auto_select_gpus": True,
        }
    else:
        dataloader_args = {"batch_size": 4}
        trainer_args = {"max_epochs": 2}

    model = LitDemucs()

    if train:
        train_dataloader = DataLoader(
            SinusoidDataset(2048, example_length=1, extend_to_valid=True), **dataloader_args
        )

        trainer = pl.Trainer(log_every_n_steps=32, default_root_dir="data/", **trainer_args)
        trainer.fit(model=model, train_dataloaders=train_dataloader)
    else:
        dataloader = DataLoader(SinusoidDataset(2, example_length=1, extend_to_valid=True))
        model = LitDemucs.load_from_checkpoint(test_checkpoint_path)

        x, y = next(iter(dataloader))
        y_pred = model.forward(x)
        y = center_trim(y, target=y_pred)

        plot_separated_with_truth(
            y.detach(),
            y_pred.detach(),
            center_trim(torch.from_numpy(dataloader.dataset.ts), target=y_pred),
        )
