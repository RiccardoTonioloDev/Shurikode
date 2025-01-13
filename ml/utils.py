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


def acc_for_prob_vec(
    device: Literal["cpu", "mps", "cuda"], pred: torch.Tensor, gt: torch.Tensor
) -> float:
    binary_pred = (pred > 0.5) * torch.ones(pred.shape).to(device)
    differences_counter = torch.sum(torch.abs(binary_pred - gt), dim=1)
    wrongs = torch.ones_like(differences_counter) * (differences_counter > 0)
    batch_size = wrongs.shape[0]
    corrects = batch_size - wrongs.sum().item().__int__()
    return corrects / batch_size


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
