import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from speechsep.cli import parse_cli_args, Args
from speechsep.dataset import LibrimixDataset
from speechsep.lightning import LitDemucs
from speechsep.mock_dataset import SinusoidDataset
from speechsep.plotting import plot_separated_with_truth
from speechsep.util import center_trim, save_as_audio


def train(args):
    dataset = _create_dataset_from_args(args)
    train_dataloader = DataLoader(dataset, **args.dataloader_args)
    checkpoint_callback = ModelCheckpoint(every_n_epochs=args.all["checkpoint_every_n_epochs"])

    trainer = pl.Trainer(
        default_root_dir="data/", callbacks=[checkpoint_callback], **args.trainer_args
    )
    trainer.fit(
        model=LitDemucs(), train_dataloaders=train_dataloader, ckpt_path=args.all["checkpoint_path"]
    )


def predict(args):
    model = LitDemucs.load_from_checkpoint(args.all["checkpoint_path"])

    dataset = _create_dataset_from_args(args)
    dataloader = DataLoader(dataset)
    ts = dataloader.dataset.ts
    dataloader = iter(dataloader)

    for _ in range(args.all["item"]):
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


def _create_dataset_from_args(args: Args):
    if args.dataset_args["dataset"] == "sinusoid":
        return SinusoidDataset.from_args(args)
    elif args.dataset_args["dataset"] == "librimix":
        return LibrimixDataset.from_args(args)


if __name__ == "__main__":
    args = parse_cli_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
