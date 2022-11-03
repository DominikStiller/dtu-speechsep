import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Dataset

from speechsep.model import valid_n_samples


class SinusoidDataset(Dataset):
    """
    A dataset of sinusoids with random amplitude, frequency and phase.

    Examples can be padded or extended to the next valid number of samples
    that is compatible with the encoder-decoder structure of the Demucs model.
    For a more detailed description, see :func:`speechsep.model.valid_n_samples`.
    """

    def __init__(
        self, n, example_length=8, sample_rate=8e3, pad_to_valid=False, extend_to_valid=False
    ):
        """
        Initialize a sinusoid dataset.

        If neither `pad_to_valid` or `extend_to_valid` is given, the number of
        samples may be invalid for Demucs.

        Args:
            n: number of examples
            example_length: length of each example [s]
            sample_rate: sample rate [Hz]
            pad_to_valid: pad with 0s to valid number of samples (for evaluation)
            extend_to_valid: extend sinusoid to valid number of samples(for training)
        """
        self.n = n
        self.random = default_rng()

        if pad_to_valid and extend_to_valid:
            raise "Cannot use both pad_to_valid and extend_to_valid"
        self.pad_to_valid = pad_to_valid
        self.extend_to_valid = extend_to_valid

        self.n_samples = example_length * sample_rate
        if pad_to_valid or extend_to_valid:
            self.n_samples_valid = valid_n_samples(self.n_samples)
        else:
            self.n_samples_valid = self.n_samples

        self.sample_rate = sample_rate
        self.ts = np.arange(0, self.n_samples_valid / sample_rate, 1 / sample_rate)
        self._ts_unpadded = np.arange(0, self.n_samples / sample_rate, 1 / sample_rate)

        # Amplitude
        self.amps = 1 + self.random.random((2, n)) * 2
        # Angular frequency
        self.omegas = 1 + self.random.random((2, n)) * 30
        # Ensure that sinusoids are below Nyquist frequency
        assert self.omegas.max().max() / (2 * np.pi) < self.sample_rate / 2
        # Initial phase
        self.phis = self.random.random((2, n)) * 2 * np.pi

    def __len__(self):
        return self.n

    def _generate_sinusoid(self, idx, speaker, ts):
        amp = self.amps[speaker, idx]
        omega = self.omegas[speaker, idx]
        phi = self.phis[speaker, idx]
        return amp * np.sin(omega * ts + phi)

    def __getitem__(self, idx):
        if self.pad_to_valid:
            speaker1 = self._generate_sinusoid(idx, 0, self._ts_unpadded)
            speaker2 = self._generate_sinusoid(idx, 1, self._ts_unpadded)

            delta = int(self.n_samples_valid - self.n_samples)
            padding_left = max(0, delta) // 2
            padding_right = delta - padding_left

            speaker1 = np.pad(speaker1, (padding_left, padding_right))
            speaker2 = np.pad(speaker2, (padding_left, padding_right))
            assert speaker1.shape[-1] == self.n_samples_valid
            assert speaker2.shape[-1] == self.n_samples_valid
        else:
            speaker1 = self._generate_sinusoid(idx, 0, self.ts)
            speaker2 = self._generate_sinusoid(idx, 1, self.ts)

        speaker1 = torch.from_numpy(speaker1).view(1, -1).float()
        speaker2 = torch.from_numpy(speaker2).view(1, -1).float()

        # Mix speaker signals
        x = speaker1 + speaker2
        # Concatenate speaker signals
        y_true = torch.cat([speaker1, speaker2], dim=0)

        return x, y_true


if __name__ == "__main__":
    dataset = SinusoidDataset(12, example_length=1, extend_to_valid=True)
    x, y = dataset[0]
    print(x.shape)
    print(y.shape)

    import matplotlib.pyplot as plt

    plt.plot(dataset.ts, x[0])
    plt.plot(dataset.ts, y[0])
    plt.plot(dataset.ts, y[1])
    plt.show()
