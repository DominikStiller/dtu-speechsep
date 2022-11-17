import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from speechsep.cli import parse_cli_args, Args
from speechsep.dataset import LibrimixDataset
from speechsep.lightning import LitDemucs
from speechsep.mock_dataset import SinusoidDataset
from speechsep.plotting import plot_separated_with_truth, save_plot
from speechsep.util import center_trim, save_as_audio


def train(args):
    train_dataset, val_dataset = _create_train_datasets_from_args(args)
    train_dataloader = DataLoader(
        train_dataset, **args.dataloader_args, persistent_workers=args["devices"] > 1
    )
    val_dataloader = DataLoader(
        val_dataset, **args.dataloader_args, persistent_workers=args["devices"] > 1
    )

    checkpoint_callback = ModelCheckpoint(every_n_epochs=args["checkpoint_every_n_epochs"])
    logger = TensorBoardLogger("data/models", name=args.dataset)
    args.save_to_json(logger.log_dir)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        **args.trainer_args,
        auto_select_gpus=True,
    )
    trainer.fit(
        model=LitDemucs(args),
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args["checkpoint_path"],
    )


def predict(args):
    print(f"Loading checkpoint {args['checkpoint_path']}...")
    model = LitDemucs.load_from_checkpoint(args["checkpoint_path"], args=args)

    print("Predicting...")
    test_dataset = _create_test_dataset_from_args(args)
    dataloader = DataLoader(test_dataset)
    ts = dataloader.dataset.ts
    dataloader = iter(dataloader)

    for _ in range(args["item"]):
        x, y = next(dataloader)
    y_pred = model.forward(x)

    x = center_trim(x, target=y_pred).detach()
    y = center_trim(y, target=y_pred).detach()
    y_pred = y_pred.detach()

    # Remove batch dimension
    x = x.squeeze(dim=0)
    y = y.squeeze(dim=0)
    y_pred = y_pred.squeeze(dim=0)

    fig = plot_separated_with_truth(
        x,
        y,
        y_pred,
        center_trim(torch.from_numpy(ts), target=y_pred),
    )
    # plt.show()

    output_path = f"data/predict/{args.dataset_args['dataset']}"
    save_plot("waveforms", output_path, fig, "png")
    save_as_audio(x, f"{output_path}/x.wav")
    save_as_audio(y, f"{output_path}/y.wav")
    save_as_audio(y_pred, f"{output_path}/y_pred.wav")


def _create_train_datasets_from_args(args: Args):
    if args.dataset_args["dataset"] == "sinusoid":
        dataset_cls = SinusoidDataset
    elif args.dataset_args["dataset"] == "librimix":
        dataset_cls = LibrimixDataset
    return dataset_cls.from_args(args, "train"), dataset_cls.from_args(args, "val")


def _create_test_dataset_from_args(args: Args):
    if args.dataset_args["dataset"] == "sinusoid":
        dataset_cls = SinusoidDataset
    elif args.dataset_args["dataset"] == "librimix":
        dataset_cls = LibrimixDataset
    return dataset_cls.from_args(args, "test")


if __name__ == "__main__":
    args = parse_cli_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
