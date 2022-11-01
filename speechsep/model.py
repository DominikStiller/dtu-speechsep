import torch
import torch.nn.functional as F
from torch import nn


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
        self.lstm = DemucsLSTM()
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
        """
        Forward-pass of the Demucs model. Only n_channels = 1 is supported.

        Args:
            x: input signal, shape (n_batch, n_channels, n_samples)

        Returns:
            Separated signal for both speakers, shape (n_batch, 2, n_samples)
        """
        skip_activations: list[torch.Tensor] = []

        for encoder in self.encoders:
            x = encoder(x)
            skip_activations.append(x)

        x = self.lstm(x)

        for decoder in self.decoders:
            skip_activation = center_trim(skip_activations.pop(), target=x)

            # x = torch.cat([x, skip_activation], dim=1)
            # Demucs adds instead of concatenates the skip activations, contrary to U-net
            x += skip_activation
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


class DemucsLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=2048, hidden_size=2048, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(4096, 2048)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # move sequence first
        x, _ = self.lstm(x)
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


def center_trim(to_trim: torch.Tensor, target: torch.Tensor, dim=-1):
    """
    Trims a tensor to match the length of another, removing equally from both sides.

    Args:
        to_trim: the tensor to trim
        target: the tensor whose length to match
        dim: the dimension in which to trim

    Returns:
        The trimmed to_trim tensor
    """
    return to_trim.narrow(dim, (to_trim.shape[dim] - target.shape[dim]) // 2, target.shape[dim])


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
    from speechsep.mock_dataset import generate_mock_data

    x, y_true = generate_mock_data(True)
    model = Demucs()

    y_pred = model.forward(x)
    y_true = center_trim(y_true, target=y_pred)
    assert y_true.shape == y_pred.shape
    print(y_true.shape)

    print(F.mse_loss(y_pred, y_true))
