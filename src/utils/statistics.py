import torch
from torch import Tensor


def calculate_frequency(data: Tensor, min_length: int = 5) -> tuple[Tensor, Tensor]:
    counts = torch.bincount(data.to(dtype=torch.int), minlength=min_length)
    empirical_probs = counts.float() / data.size(0)
    values = torch.arange(len(empirical_probs))
    return values, empirical_probs
