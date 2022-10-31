import torch
import matplotlib.pyplot as plt
import seaborn as sb


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
