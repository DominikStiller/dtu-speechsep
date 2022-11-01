import numpy as np
import torch
import torch.nn.functional as F

from speechsep.model import valid_length


def generate_mock_data(pad_to_valid=False):

    example_length = 8
    sample_rate = 8e3

    ts = np.arange(0, example_length, 1 / sample_rate)
    speaker1 = np.sin(5 * ts)
    speaker2 = np.sin(11 * ts)

    speaker1 = torch.from_numpy(speaker1).view(1, 1, len(ts)).float()
    speaker2 = torch.from_numpy(speaker2).view(1, 1, len(ts)).float()

    if pad_to_valid:
        target_n_samples = int(valid_length(len(ts)))
        padding = max(0, target_n_samples - len(ts))

        speaker1 = F.pad(speaker1, (0, padding))
        speaker2 = F.pad(speaker2, (0, padding))

    # Mix speaker signals
    x = speaker1 + speaker2
    # Concatenate speaker signals
    y_true = torch.cat([speaker1, speaker2], dim=1)

    return x, y_true
