import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import torch

# Initialize seaborn formatting once module is loaded
sb.set(
    context="paper",
    style="ticks",
    font_scale=1.6,
    font="sans-serif",
    rc={
        "lines.linewidth": 1.2,
        "axes.titleweight": "bold",
    },
)


def plot_waveform(waveform, sample_rate):
    # Assume mono signal
    waveform = waveform.numpy()[0]

    _, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    plt.plot(time_axis, waveform)
    plt.show(block=False)


def plot_specgram(waveform, sample_rate):
    waveform = waveform.numpy()[0]

    plt.specgram(waveform, Fs=sample_rate)
    plt.show(block=False)


def plot_separated_with_truth(
    x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor, ts: np.ndarray
):
    """
    Plot the separated signal on top of the ground truth.

    Args:
        x: mixed signal, shape (n_channels, n_samples)
        y: ground truth, shape (n_channels, n_samples)
        y_pred: separated signal, shape (n_channels, n_samples)
        ts: time steps
    """
    assert len(x.shape) == 2 and x.shape[0] == 1
    assert len(y.shape) == 2 and y.shape[0] == 2
    assert len(y_pred.shape) == 2 and y_pred.shape[0] == 2

    fig, axs = plt.subplots(3, 1, figsize=(6, 8), tight_layout=True)

    ax = axs[0]
    ax.plot(ts, x[0])
    ax.set_title("Mixed")

    ax = axs[1]
    ax.plot(ts, y_pred[0], label="Prediction")
    ax.plot(ts, y[0], label="Ground truth")
    ax.set_title("Speaker 1")
    ax.legend()

    ax = axs[2]
    ax.plot(ts, y_pred[1], label="Prediction")
    ax.plot(ts, y[1], label="Ground truth")
    ax.set_title("Speaker 2")
    ax.legend()

    plt.show()
