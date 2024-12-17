import torch


def loss_function(pred: torch.Tensor, gt: torch.Tensor):
    return torch.mean(torch.sum((pred - gt) ** 2, 1))
