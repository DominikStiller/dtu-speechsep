import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim

from speechsep.model import Demucs
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
    import matplotlib.pyplot as plt
    import os

    train = False
    is_hpc = "LSF_ENVDIR" in os.environ
    use_gpu = is_hpc and torch.cuda.is_available()
    checkpoint_path = "data/lightning_logs/version_4/checkpoints/epoch=19-step=1280.ckpt"

    dataloader_args = {}
    trainer_args = {}
    if use_gpu:
        dataloader_args = {"num_workers": 4, "persistent_workers": True}
        trainer_args = {
            "accelerator": "gpu",
            "devices": 1,
            "auto_select_gpus": True,
        }

    model = LitDemucs()

    if train:
        train_dataloader = DataLoader(
            SinusoidDataset(2048, example_length=1, extend_to_valid=True),
            batch_size=16,
            **dataloader_args
        )

        trainer = pl.Trainer(
            max_epochs=20, log_every_n_steps=32, default_root_dir="data/", **trainer_args
        )
        trainer.fit(model=model, train_dataloaders=train_dataloader)
    else:
        dataloader = DataLoader(SinusoidDataset(1, example_length=1, pad_to_valid=True))
        model.load_from_checkpoint(checkpoint_path)

        x, y = next(iter(dataloader))
        y_pred = model.forward(x)
        y = center_trim(y, target=y_pred)

        y = y.detach()
        y_pred = y_pred.detach()

        fig, axs = plt.subplots(2, 1)
        ts = center_trim(torch.from_numpy(dataloader.dataset.ts), target=y_pred)

        ax = axs[0]
        ax.plot(ts, y_pred[0, 0], label="Prediction")
        ax.plot(ts, y[0, 0], label="Ground truth")
        ax.set_title("Speaker 1")
        ax.legend()

        ax = axs[1]
        ax.plot(ts, y_pred[0, 1], label="Prediction")
        ax.plot(ts, y[0, 1], label="Ground truth")
        ax.set_title("Speaker 2")
        ax.legend()

        plt.tight_layout()
        plt.show()
