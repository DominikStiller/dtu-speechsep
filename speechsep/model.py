import torch
from torch import nn
import torch.nn.functional as F


class Demucs(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                DemucsEncoder(1, 64),
                DemucsEncoder(64, 128),
                DemucsEncoder(128, 256),
                DemucsEncoder(256, 512),
                DemucsEncoder(512, 1024),
                DemucsEncoder(1024, 2048),
            ]
        )
        self.blstm = BilinearLSTM()
        self.decoders = nn.ModuleList(
            [
                DemucsDecoder(2048, 1024),
                DemucsDecoder(1024, 512),
                DemucsDecoder(512, 256),
                DemucsDecoder(256, 128),
                DemucsDecoder(128, 64),
                DemucsDecoder(64, 2, use_activation=False),
            ]
        )

    def forward(self, x):
        skip_activations: list[torch.Tensor] = []

        for encoder in self.encoders:
            x = encoder(x)
            skip_activations.append(x)

        x = self.blstm(x)

        for decoder in self.decoders:
            skip_activation = skip_activations.pop()

            # Center-trim skip activation to match x
            skip_activation = skip_activation.narrow(
                -1, (skip_activation.shape[-1] - x.shape[-1]) // 2, x.shape[-1]
            )

            x = torch.cat([x, skip_activation])
            x = decoder(x)

        return x


class DemucsEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv1d(out_channels, 2 * out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.glu(x, dim=1)  # split in channel dimension
        return x


class BilinearLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=2048, hidden_size=2048, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(4096, 2048)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # move sequence first
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x


class DemucsDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_activation=True):
        super().__init__()

        self.conv_1 = nn.Conv1d(in_channels, 2 * in_channels, kernel_size=3, stride=1)
        self.conv_2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=8, stride=4)
        self.use_activation = use_activation

    def forward(self, x):
        x = self.conv_1(x)
        x = F.glu(x, dim=1)  # split in channel dimension
        x = self.conv_2(x)
        if self.use_activation:
            x = F.relu(x)
        return x


def _generate_test_data():
    import numpy as np

    example_length = 8
    sample_rate = 8e3

    ts = np.arange(0, example_length, 1 / sample_rate)
    speaker1 = np.sin(5 * ts)
    speaker2 = np.sin(11 * ts)

    speaker1 = torch.from_numpy(speaker1).view(1, 1, len(ts)).float()
    speaker2 = torch.from_numpy(speaker2).view(1, 1, len(ts)).float()

    target_n_samples = int(valid_length(len(ts)) + sample_rate)
    padding = max(0, target_n_samples - len(ts))

    # speaker1 = F.pad(speaker1, (0, padding))
    # speaker2 = F.pad(speaker2, (0, padding))

    x = speaker1 + speaker2
    y = torch.vstack([speaker1, speaker2])

    return x, y


# From https://github.com/facebookresearch/demucs/blob/v2/demucs/model.py#L145
def valid_length(length):
    import math

    resample = False
    depth = 6
    kernel_size = 8
    stride = 4
    context = 3

    if resample:
        length *= 2
    for _ in range(depth):
        length = math.ceil((length - kernel_size) / stride) + 1
        length = max(1, length)
        length += context - 1
    for _ in range(depth):
        length = (length - 1) * stride + kernel_size

    if resample:
        length = math.ceil(length / 2)
    return int(length)


if __name__ == "__main__":
    x, y = _generate_test_data()
    model = Demucs()

    x = model.forward(x)
    print(x.shape)
