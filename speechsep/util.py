from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


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


def pad(to_pad: Union[np.ndarray, torch.Tensor], target_length: int):
    delta = int(target_length - to_pad.shape[-1])
    padding_left = max(0, delta) // 2
    padding_right = delta - padding_left

    if isinstance(to_pad, np.ndarray):
        return np.pad(to_pad, (padding_left, padding_right))
    elif isinstance(to_pad, torch.Tensor):
        return F.pad(to_pad, (padding_left, padding_right))
