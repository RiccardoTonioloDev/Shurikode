import torch


def loss_function(pred: torch.Tensor, gt: torch.Tensor):
    return torch.mean(torch.sum(torch.abs(pred - gt) ** 2, 1))
