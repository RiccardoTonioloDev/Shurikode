import torch


def mse_loss_function(pred: torch.Tensor, gt: torch.Tensor):
    return torch.mean(torch.sum((pred - gt) ** 2, 1))


def xentr_loss_function(pred: torch.Tensor, gt: torch.Tensor):
    return torch.mean(
        torch.sum(
            -(pred * torch.log(gt) + (1 - pred) * torch.log(1 - gt)),
            1,
        )
    )
