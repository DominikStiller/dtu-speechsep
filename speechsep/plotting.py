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


def plot_separated_with_truth(y: torch.Tensor, y_pred: torch.Tensor, ts: np.ndarray, idx=0):
    """
    Plot the separated signal on top of the ground truth.

    Args:
        y: ground truth, shape (n_batch, n_channels, n_samples)
        y_pred: separated signal, shape (n_batch, n_channels, n_samples)
        ts: time steps
        idx: example index within batch

    Returns:

    """
    fig, axs = plt.subplots(2, 1, tight_layout=True)

    ax = axs[0]
    ax.plot(ts, y_pred[idx, 0], label="Prediction")
    ax.plot(ts, y[idx, 0], label="Ground truth")
    ax.set_title("Speaker 1")
    ax.legend()

    ax = axs[1]
    ax.plot(ts, y_pred[idx, 1], label="Prediction")
    ax.plot(ts, y[idx, 1], label="Ground truth")
    ax.set_title("Speaker 2")
    ax.legend()

    plt.show()
