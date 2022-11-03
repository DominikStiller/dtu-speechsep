import torch


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
