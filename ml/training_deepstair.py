from model import BinNet_deepstair
from dataset import shurikode_dataset
from losses import mse_loss_function
from torch.optim import Adam
from tqdm import tqdm
from utils import (
    save_model,
    number_of_correct_predictions,
    avg_errors_in_wrong_predictions,
)
from typing import List

import torch
import argparse
import wandb

parser = argparse.ArgumentParser(
    description="Arguments for the training procedure of the Shurikode decoder model."
)
parser.add_argument("--exp_name", type=str, help="The name of the current experiment.")
parser.add_argument(
    "--checkpoints_dir",
    type=str,
    help="The directory that will be containing the checkpoints.",
)
parser.add_argument(
    "--datasets_path",
    type=str,
    help="The directory where the train and val datasets directories are located.",
)
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    print("TRAINING ON CUDA")
elif torch.mps.is_available():
    device = "mps"
    print("TRAINING ON MPS")


m = BinNet_deepstair().to(device)

epochs_n = 90
lr = 1e-4
train_variety = 400
val_variety = 30
batch_size = 32

optimizer = Adam(m.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

wandb.init(
    project="Shurikode_decoder",
    name=args.exp_name,
    config={
        "num_epochs": epochs_n,
        "learning_rate": lr,
    },
)

is_first_epoch = True
min_loss = torch.Tensor([0]).to(device)
checkpoint_files: List[str] = []
train_dataset = shurikode_dataset(
    data_path=args.datasets_path, type="train", variety=train_variety
)
train_dataloader = train_dataset.make_dataloader(batch_size=batch_size)
val_dataset = shurikode_dataset(
    data_path=args.datasets_path, type="val", variety=val_variety
)
val_dataloader = val_dataset.make_dataloader(batch_size=batch_size)

for epoch in range(epochs_n):

    ########################################## TRAINING EPOCH ##########################################
    m.train()
    tdataloader = tqdm(train_dataloader, unit="batch")
    for i, (img, gt) in enumerate(tdataloader):
        tdataloader.set_description(f"(training) Epoch {epoch}/{epochs_n}")
        img, gt = img.to(device), gt.to(device)

        pred: torch.Tensor = m(img)

        loss: torch.Tensor = mse_loss_function(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_04 = number_of_correct_predictions(device, pred, gt, 0.4) / batch_size
        avg_err_04 = avg_errors_in_wrong_predictions(device, pred, gt, 0.4)
        acc_05 = number_of_correct_predictions(device, pred, gt, 0.5) / batch_size
        avg_err_05 = avg_errors_in_wrong_predictions(device, pred, gt, 0.5)
        acc_08 = number_of_correct_predictions(device, pred, gt, 0.8) / batch_size
        avg_err_08 = avg_errors_in_wrong_predictions(device, pred, gt, 0.8)

        tdataloader.set_postfix(
            loss=loss.item(),
            acc_04=acc_04,
            acc_05=acc_05,
            acc_08=acc_08,
            avg_err_04=avg_err_04,
            avg_err_05=avg_err_05,
            avg_err_08=avg_err_08,
        )

        tdataloader.set_postfix(
            loss=loss.item(), acc_04=acc_04, acc_05=acc_05, acc_08=acc_08
        )
        wandb.log(
            {
                "loss": loss.item(),
                "acc_04": acc_04,
                "acc_05": acc_05,
                "acc_08": acc_08,
                "avg_err_04": avg_err_04,
                "avg_err_05": avg_err_05,
                "avg_err_08": avg_err_08,
            }
        )

    ############################################ VALIDATION ###########################################
    tdataloader = tqdm(val_dataloader, unit="batch")
    loss_tower = []
    with torch.no_grad():
        m.eval()
        for img, gt in tdataloader:
            tdataloader.set_description(f"(validation) Epoch {epoch}/{epochs_n}")
            img, gt = img.to(device), gt.to(device)

            pred: torch.Tensor = m(img)

            # loss: torch.Tensor = mse_loss_function(pred, gt)
            loss: torch.Tensor = mse_loss_function(pred, gt)
            loss_tower.append(loss)

            acc_04 = number_of_correct_predictions(device, pred, gt, 0.4) / batch_size
            avg_err_04 = avg_errors_in_wrong_predictions(device, pred, gt, 0.4)
            acc_05 = number_of_correct_predictions(device, pred, gt, 0.5) / batch_size
            avg_err_05 = avg_errors_in_wrong_predictions(device, pred, gt, 0.5)
            acc_08 = number_of_correct_predictions(device, pred, gt, 0.8) / batch_size
            avg_err_08 = avg_errors_in_wrong_predictions(device, pred, gt, 0.8)

            tdataloader.set_postfix(
                loss=loss.item(),
                acc_04=acc_04,
                acc_05=acc_05,
                acc_08=acc_08,
                avg_err_04=avg_err_04,
                avg_err_05=avg_err_05,
                avg_err_08=avg_err_08,
            )

checkpoint_filename = save_model(
    args.checkpoints_dir,
    f"final_{args.exp_name}",
    m.state_dict(),
)
