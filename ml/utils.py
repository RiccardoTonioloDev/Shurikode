from typing import Dict, Any, Literal

import torch
import os


def save_model(
    checkpoints_path: str, name: str, model_state_dict: Dict[str, Any]
) -> str:
    checkpoint_filename = os.path.join(checkpoints_path, f"checkpoint_{name}.pth.tar")
    torch.save(model_state_dict, checkpoint_filename)
    return checkpoint_filename


def number_of_correct_predictions(
    device: Literal["cpu", "mps", "cuda"],
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold=0.5,
) -> int:
    binary_pred = (pred > threshold) * torch.ones(pred.shape).to(
        device
    )  # 1 where there is a difference
    tensor_of_differences = torch.sum(
        torch.abs(binary_pred - gt), dim=1
    )  # computing the number of differences per image
    tensor_of_correctness = (
        tensor_of_differences > 0
    ) * torch.ones(  # 1 where the image was mislabeled
        tensor_of_differences.shape
    ).to(
        device
    )
    number_of_wrong = torch.sum(
        tensor_of_correctness
    )  # computing how many images where mislabeled
    return (
        (len(tensor_of_correctness) - number_of_wrong).item().__int__()
    )  # computing how many images where right


def avg_errors_in_wrong_predictions(
    device: Literal["cpu", "mps", "cuda"],
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold=0.5,
) -> float:
    binary_pred = (pred > threshold) * torch.ones(pred.shape).to(
        device
    )  # 1 where there is a difference
    tensor_of_differences = torch.sum(
        torch.abs(binary_pred - gt), dim=1
    )  # computing the number of differences per image

    tensor_of_correctness = (
        tensor_of_differences > 0
    ) * torch.ones(  # 1 where the image was mislabeled
        tensor_of_differences.shape
    ).to(
        device
    )

    number_of_wrong = torch.sum(
        tensor_of_correctness
    )  # computing how many images where mislabeled

    avg_errors_per_image = torch.sum(
        tensor_of_differences
    )  # computing how many images where mislabeled
    return (
        avg_errors_per_image.item().__int__() / number_of_wrong.item().__int__()
    )  # computing how many images where right
