from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from typing import Literal
from utils import avg_errors_in_wrong_predictions, number_of_correct_predictions
from model import Create_ResNet50
from dataset import shurikode_dataset, shurikode_dataset_hamming

import torch
import os


def validate(
    m: nn.Module,
    val_dataloader: DataLoader,
    device: Literal["cuda", "mps", "cpu"],
    batch_size: int,
):
    tdataloader = tqdm(val_dataloader, unit="batch")
    acc_04 = 0
    sum_err_04 = 0
    acc_05 = 0
    sum_err_05 = 0
    acc_08 = 0
    sum_err_08 = 0
    errors_04 = 0
    errors_05 = 0
    errors_08 = 0
    with torch.no_grad():
        m.eval()
        for i, (img, gt) in enumerate(tdataloader):
            img, gt = img.to(device), gt.to(device)

            pred: torch.Tensor = torch.sigmoid(m(img))

            acc_04 = (
                (acc_04 * i)
                + (number_of_correct_predictions(device, pred, gt, 0.4) / batch_size)
            ) / (i + 1)
            err_04 = avg_errors_in_wrong_predictions(device, pred, gt, 0.4)
            sum_err_04 += err_04
            if err_04 > 0:
                errors_04 += 1

            acc_05 = (
                (acc_05 * i)
                + number_of_correct_predictions(device, pred, gt, 0.5) / batch_size
            ) / (i + 1)
            err_05 = avg_errors_in_wrong_predictions(device, pred, gt, 0.5)
            sum_err_05 += err_05
            if err_05 > 0:
                errors_05 += 1

            acc_08 = (
                (acc_08 * i)
                + number_of_correct_predictions(device, pred, gt, 0.8) / batch_size
            ) / (i + 1)
            err_08 = avg_errors_in_wrong_predictions(device, pred, gt, 0.8)
            sum_err_08 += err_08
            if err_08 > 0:
                errors_08 += 1

            tdataloader.set_postfix(
                acc_04=acc_04,
                acc_05=acc_05,
                acc_08=acc_08,
                avg_err_04=sum_err_04 / (errors_04 if errors_04 != 0 else 1),
                avg_err_05=sum_err_05 / (errors_05 if errors_05 != 0 else 1),
                avg_err_08=sum_err_08 / (errors_08 if errors_08 != 0 else 1),
            )


if __name__ == "__main__":
    """
    # TO EXECUTE AS MAIN ONLY IN CLUSTER (VERIFY TO HAVE VALID PATHS FOR CHECKPOINTS)
    """
    print("----------------------------- NORMAL -----------------------------")
    print("")

    checkpoints_dir = "/home/rtoniolo/Shurikode/ml/checkpoints"
    checkpoints_to_use = "checkpoint_e020_ResNet50_sig.pth.tar"

    datasets_dir = "/home/rtoniolo/Datasets/Shurikode"
    device = "cuda"

    state_dict = torch.load(os.path.join(checkpoints_dir, checkpoints_to_use))
    m = Create_ResNet50(state_dict, 8).to(device)
    dataloader = shurikode_dataset(datasets_dir, "val", 30).make_dataloader(1, False)

    validate(m, dataloader, device, 1)

    print("")
    print("----------------------------- HAMMING -----------------------------")

    checkpoints_to_use = "checkpoint_e089_ResNet50_hamming_sig.pth.tar"

    state_dict = torch.load(os.path.join(checkpoints_dir, checkpoints_to_use))
    m = Create_ResNet50(state_dict, 12).to(device)
    dataloader = shurikode_dataset(datasets_dir, "val", 30).make_dataloader(1, False)

    validate(m, dataloader, device, 1)
