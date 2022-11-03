import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Dataset

from speechsep.model import valid_n_samples


class SinusoidDataset(Dataset):
    def __init__(
        self, n, example_length=8, sample_rate=8e3, pad_to_valid=False, extend_to_valid=False
    ):
        """

        Args:
            n:
            example_length:
            sample_rate:
            pad_to_valid: Use for evaluation
            extend_to_valid: Use for training
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

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self.random = default_rng(24)
        amp1 = 1 + self.random.random() * 2
        amp2 = 1 + self.random.random() * 2
        omega1 = 1 + self.random.random() * 30
        omega2 = 1 + self.random.random() * 30
        phi1 = self.random.random() * 2 * np.pi
        phi2 = self.random.random() * 2 * np.pi

        # Ensure that sinusoids are below Nyquist frequency
        assert omega1 / (2 * np.pi) < self.sample_rate / 2
        assert omega2 / (2 * np.pi) < self.sample_rate / 2

        if self.pad_to_valid:
            speaker1 = amp1 * np.sin(omega1 * self._ts_unpadded + phi1)
            speaker2 = amp2 * np.sin(omega2 * self._ts_unpadded + phi2)

            delta = int(self.n_samples_valid - self.n_samples)
            padding_left = max(0, delta) // 2
            padding_right = delta - padding_left

            speaker1 = np.pad(speaker1, (padding_left, padding_right))
            speaker2 = np.pad(speaker2, (padding_left, padding_right))
            assert speaker1.shape[-1] == self.n_samples_valid
            assert speaker2.shape[-1] == self.n_samples_valid
        else:
            speaker1 = amp1 * np.sin(omega1 * self.ts + phi1)
            speaker2 = amp2 * np.sin(omega2 * self.ts + phi2)

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
