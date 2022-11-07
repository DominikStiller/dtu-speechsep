import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim

from speechsep.dataset import LibrimixDataset
from speechsep.model import Demucs
from speechsep.plotting import plot_separated_with_truth
from speechsep.util import center_trim, save_as_audio

from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


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
        loss = F.l1_loss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pesq, pesq0, pesq1 = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss, "val_pesq": pesq, "val_pesq_0": pesq0, "val_pesq_1": pesq1}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, pesq, pesq0, pesq1 = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_loss": loss, "test_pesq": pesq, "test_pesq_0": pesq0, "test_pesq_1": pesq1}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y = center_trim(y, target=y_pred)
        loss = F.l1_loss(y_pred, y)
        # calculate pesq
        nb_pesq = PerceptualEvaluationSpeechQuality(8000, 'nb')
        pesq = nb_pesq(y_pred, y)
        pesq0 = nb_pesq(y_pred[:, 0:1, :], y[:, 0:1, :])
        pesq1 = nb_pesq(y_pred[:, 1:2, :], y[:, 1:2, :])
        return loss, pesq, pesq0, pesq1

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from pytorch_lightning.callbacks import ModelCheckpoint
    import os

    train = False
    use_gpu = False

    # train_checkpoint_path = None
    train_checkpoint_path = "data/lightning_logs/version_0/checkpoints/epoch=4-step=20.ckpt"
    # train_dataset = SinusoidDataset(4096 * 8, example_length=1, extend_to_valid=True)
    train_dataset = LibrimixDataset(
        "data/datasets/mini/mixture_mini_mix_both.csv", pad_to_valid=True
    )

    val_dataset = LibrimixDataset(
        "data/datasets/mini/mixture_mini_mix_both.csv", pad_to_valid=True
    )

    # test_checkpoint_path = "data/lightning_logs/version_19/checkpoints/epoch=119-step=30720.ckpt"  # Sinusoid
    # test_checkpoint_path = (
    #     "data/lightning_logs/version_15/checkpoints/epoch=499-step=1000.ckpt"  # LibriMix mini, 500 epochs
    # )
    test_checkpoint_path = "data/lightning_logs/version_0/checkpoints/epoch=3-step=20.ckpt"  # LibriMix mini, 4 epochs
    # test_dataset = SinusoidDataset(50, example_length=1, extend_to_valid=True, seed=45)
    test_dataset = LibrimixDataset(
        "data/datasets/mini/mixture_mini_mix_both.csv", pad_to_valid=True
    )

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
        dataloader_args = {"batch_size": 64, "num_workers": 4, "persistent_workers": True}
        trainer_args = {
            "max_epochs": 2500,
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
        train = True
        dataloader_args = {"batch_size": 32}
        trainer_args = {"max_epochs": 1, "log_every_n_steps": 4}

    validation_every_step = 4

    if train:
        train_dataloader = DataLoader(train_dataset, **dataloader_args)
        val_dataloader = DataLoader(val_dataset, **dataloader_args)
        checkpoint_callback = ModelCheckpoint(every_n_epochs=1)

        trainer = pl.Trainer(
            default_root_dir="data/", check_val_every_n_epoch=None, val_check_interval=validation_every_step,
            callbacks=[checkpoint_callback], **trainer_args
        )
        trainer.fit(
            model=LitDemucs(), train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )

        test_dataloader = DataLoader(test_dataset)
        trainer.test(dataloaders=test_dataloader, ckpt_path="best")

    else:
        model = LitDemucs.load_from_checkpoint(test_checkpoint_path)

        dataloader = DataLoader(test_dataset)
        ts = dataloader.dataset.ts
        dataloader = iter(dataloader)

        for _ in range(24):
            x, y = next(dataloader)
        y_pred = model.forward(x)

        x = center_trim(x, target=y_pred).detach()
        y = center_trim(y, target=y_pred).detach()
        y_pred = y_pred.detach()

        # Remove batch dimension
        x = x.squeeze(dim=0)
        y = y.squeeze(dim=0)
        y_pred = y_pred.squeeze(dim=0)

        plot_separated_with_truth(
            x,
            y,
            y_pred,
            center_trim(torch.from_numpy(ts), target=y_pred),
        )

        save_as_audio(x, "data/predict/x.wav")
        save_as_audio(y, "data/predict/y.wav")
        save_as_audio(y_pred, "data/predict/y_pred.wav")
