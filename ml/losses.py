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


def xentr_loss_function_stair(
    pred_complete: torch.Tensor,
    pred_4: torch.Tensor,
    pred_3: torch.Tensor,
    pred_2: torch.Tensor,
    pred_1: torch.Tensor,
    gt: torch.Tensor,
):
    return (
        torch.mean(
            torch.sum(
                -(
                    pred_complete * torch.log(gt)
                    + (1 - pred_complete) * torch.log(1 - gt)
                ),
                1,
            )
        )
        + torch.mean(
            torch.sum(
                -(pred_4 * torch.log(gt) + (1 - pred_4) * torch.log(1 - gt)),
                1,
            )
        )
        + torch.mean(
            torch.sum(
                -(pred_3 * torch.log(gt) + (1 - pred_3) * torch.log(1 - gt)),
                1,
            )
        )
        + torch.mean(
            torch.sum(
                -(pred_2 * torch.log(gt) + (1 - pred_2) * torch.log(1 - gt)),
                1,
            )
        )
        + torch.mean(
            torch.sum(
                -(pred_1 * torch.log(gt) + (1 - pred_1) * torch.log(1 - gt)),
                1,
            )
        )
    )
