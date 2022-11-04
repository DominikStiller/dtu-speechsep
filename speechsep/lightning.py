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
    from pytorch_lightning.callbacks import ModelCheckpoint
    import os

    train = False
    use_gpu = False

    # train_checkpoint_path = None
    train_checkpoint_path = "data/lightning_logs/version_12/checkpoints/epoch=49-step=15650.ckpt"
    test_checkpoint_path = "data/lightning_logs/version_10/checkpoints/epoch=49-step=1600.ckpt"
    # test_checkpoint_path = "data/lightning_logs/version_14/checkpoints/epoch=99-step=200.ckpt"

    # Decide whether execution is on HPC node and if GPU should be used
    is_hpc = "LSF_ENVDIR" in os.environ
    if is_hpc:
        train = True
        # On HPC nodes, always use GPU if available
        use_gpu = torch.cuda.is_available()
    else:
        # On other servers/computers, use GPU only if desired and available
        use_gpu = use_gpu and torch.cuda.is_available()

    # Determine training configuration based on HPC/GPU
    if is_hpc and use_gpu:
        dataloader_args = {"batch_size": 16, "num_workers": 4, "persistent_workers": True}
        trainer_args = {
            "max_epochs": 150,
            "log_every_n_steps": 10,
            "accelerator": "gpu",
            "devices": 2,
            "auto_select_gpus": True,
        }
    elif use_gpu:
        dataloader_args = {"batch_size": 16, "num_workers": 4}
        trainer_args = {
            "max_epochs": 50,
            "log_every_n_steps": 32,
            "accelerator": "gpu",
        }
    else:
        dataloader_args = {"batch_size": 1}
        trainer_args = {"max_epochs": 100, "log_every_n_steps": 5}

    if train:
        train_dataloader = DataLoader(
            SinusoidDataset(10000, example_length=1, extend_to_valid=True), **dataloader_args
        )
        checkpoint_callback = ModelCheckpoint(every_n_epochs=5)

        trainer = pl.Trainer(
            default_root_dir="data/", callbacks=[checkpoint_callback], **trainer_args
        )
        trainer.fit(
            model=LitDemucs(), train_dataloaders=train_dataloader, ckpt_path=train_checkpoint_path
        )
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
