# Similar to the methods used in: (A. Défossez, N. Usunier, L. Bottou, and F. Bach, Music source separation in the waveform domain, 2019. doi)
# 1. Ablation study:
#   a) no BiLSTM
#   b) no pitch/tempo aug.
#   c) no initial weight rescaling
#   d) no 1x1 convolutions in encoder
#   e) ReLU instead of GLU
#   f) no BiLSTM, depth=7
#   g) MSE loss
#   h) no resampling
#   i) no shift trick
#   j) kernel size of 1 in decoder convolutions
#
# 2. Change the initial number of channels on the model size: [32, 48, 64], w/wo DiffQ quantized
#
# Adjust the model one by one, train the models and compare test results.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy.typing import ArrayLike
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from speechsep.plotting import format_plot, save_plot

METRIC_NAMES = {
    "train_loss": "Training loss",
    "val_loss": "Validation loss",
    "val_pesq": "Validation PESQ",
    "val_stoi": "Validation STOI",
    "val_sisdr": "Validation SI-SDR",
}


def plot_training(log: pd.DataFrame, metrics: list[str], params: dict):
    fig, (ax_loss, ax_metrics) = plt.subplots(2, 1, figsize=(12, 8), sharex="all")

    for metric in metrics:
        if metric not in METRIC_NAMES.keys():
            continue

        if "loss" in metric:
            ax = ax_loss
        else:
            ax = ax_metrics

        rows = log.dropna(subset=metric)[["step", metric]]
        with pd.option_context("mode.chained_assignment", None):
            # Smooth with rolling window proportional to logging frequency
            rolling_window = max(1, int(np.log1p(_get_metric_frequency(log, metric)) * 200))
            rows[metric] = rows[metric].rolling(rolling_window, center=True).mean()

        ax.plot(rows["step"], rows[metric], label=METRIC_NAMES[metric])

    ax_metrics.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (lower is better)")
    ax_metrics.set_ylabel("Metric (higher is better)")

    ax_loss.legend()
    ax_metrics.legend()

    ax_metrics.set_xticks(*_get_ticks(log))

    plt.suptitle(f"Training on {params['dataset']} (Version {params['version']})")

    format_plot()
    plt.show()
    save_plot("training", params["plot_folder"], fig)


def _get_metric_frequency(log: pd.DataFrame, metric: str) -> int:
    """Get the frequency at which the metric is logged."""
    steps_between_logs = log[log[metric].notna()]["step"].diff().mode().iloc[0]
    return 1 / steps_between_logs


def _get_ticks(log: pd.DataFrame) -> tuple[ArrayLike, ArrayLike]:
    """Get epoch ticks and labels for the x-axis of the plot."""
    # Find optimal ticks based on MaxNLocator
    # Settings for MaxNLocator correspond to AutoLocator but with pruning
    target_steps = MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10], prune="upper").tick_values(
        log["step"].min(), log["step"].max()
    )
    # Find steps that are closest to target steps
    closest_steps = pd.merge_asof(
        pd.Series(target_steps, name="step_tick").astype(int),
        log,
        left_on="step_tick",
        right_on="step",
        direction="nearest",
    )
    return closest_steps["step"], closest_steps["epoch"]


def load_log(folder: Path):
    file = str(next(folder.glob("events.out.*")))

    log = []

    # Load events from log file
    event_acc = EventAccumulator(file).Reload()
    for metric in event_acc.Tags()["scalars"]:
        for event in event_acc.Scalars(metric):
            log.append(
                {
                    "step": event.step,
                    "metric": metric,
                    "value": event.value,
                }
            )

    log = pd.DataFrame.from_records(log)
    metrics = list(log["metric"].unique())
    metrics.remove("epoch")

    # Remove non-first rows with multiple values for same metric at a given step
    log = log.groupby(["metric", "step"]).first().reset_index()
    # Make each metric a column
    log = log.pivot(index="step", columns="metric", values="value").reset_index()
    # Drop rows with no values for epoch
    log = log.dropna(subset="epoch").reset_index(drop=True)
    log["epoch"] = log["epoch"].astype(int)

    return log, metrics


if __name__ == "__main__":
    model_folder = Path(sys.argv[1])

    params = {
        "plot_folder": model_folder / "Path",
        "version": model_folder.parts[-1].replace("version_", ""),
        "dataset": model_folder.parts[-2],
    }

    log, metrics = load_log(model_folder)
    plot_training(log, metrics, params)
