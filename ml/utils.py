from typing import Dict, Any, Literal, List, Tuple

import torchvision.transforms.v2 as T
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
    return avg_errors_per_image.item().__int__() / (
        number_of_wrong.item().__int__() if number_of_wrong.item().__int__() > 0 else 1
    )  # computing how many images where right


def generate_hamming(bits: List[int]) -> List[int]:
    # Calcolo il numero di bit di parità necessari
    n = len(bits)
    m = 0
    while (2**m) < (n + m + 1):
        m += 1

    # Positioning parity bits in the list
    hamming = []
    j = 0
    k = 0
    for i in range(1, n + m + 1):
        if i == 2**j:  # Positioning a parity bit
            hamming.append(0)
            j += 1
        else:
            hamming.append(bits[k])
            k += 1

    # Calculating parity bits values
    for p in range(m):
        parity_pos = 2**p
        parity = 0
        for i in range(1, len(hamming) + 1):
            if i & parity_pos:  # Verifying if bit is part of the group
                parity ^= hamming[i - 1]
        hamming[parity_pos - 1] = parity

    return hamming


def detect_and_correct(hamming: List[int]) -> Tuple[List[int], int]:
    m = 0
    while (2**m) < len(hamming):
        m += 1

    # Calculating error position
    error_pos = 0
    for p in range(m):
        parity_pos = 2**p
        parity = 0
        for i in range(1, len(hamming) + 1):
            if i & parity_pos:
                parity ^= hamming[i - 1]
        if parity != 0:
            error_pos += parity_pos

    # Correcting the error if necessary
    if error_pos > 0:
        hamming[error_pos - 1] ^= 1

    # Removing parity bits to return only the data
    original_bits = []
    j = 0
    for i in range(1, len(hamming) + 1):
        if i != 2**j:
            original_bits.append(hamming[i - 1])
        else:
            j += 1

    return original_bits, error_pos


def padding_to_counter_ResNet_center_crop(image_tensor, target_size):
    """
    Shrinks the image while keeping the original tensor dimensions by adding a black background.

    Args:
        image_tensor (torch.Tensor): Input tensor of shape (C, H, W).
        target_size (tuple): Desired size for the image (target_height, target_width).

    Returns:
        torch.Tensor: Tensor of the original size with the shrunk image centered and black background.
    """
    # Original dimensions
    _, orig_height, orig_width = image_tensor.shape

    # Resize the image to the target size
    resize = T.Resize(target_size)
    resized_image = resize(image_tensor)

    # Create a black background tensor of the original size
    black_background = torch.zeros_like(image_tensor)

    # Calculate top-left corner for placing the resized image
    target_height, target_width = target_size
    start_y = (orig_height - target_height) // 2
    start_x = (orig_width - target_width) // 2

    # Place the resized image onto the black background
    black_background[
        :, start_y : start_y + target_height, start_x : start_x + target_width
    ] = resized_image

    return black_background


if __name__ == "__main__":
    hamming = generate_hamming([0, 1, 0, 1, 1, 0, 1, 0])
    print(hamming)
    data, err = detect_and_correct(hamming)
    print(data, err)
    hamming[4] = 1 if hamming[4] == 0 else 0
    data, err = detect_and_correct(hamming)
    print(data, err)
    hamming[7] = 1 if hamming[7] == 0 else 0
    data, err = detect_and_correct(hamming)
    print(data, err)
    hamming[1] = 1 if hamming[1] == 0 else 0
    data, err = detect_and_correct(hamming)
    print(data, err)
    hamming[0] = 1 if hamming[0] == 0 else 0
    data, err = detect_and_correct(hamming)
    print(data, err)
