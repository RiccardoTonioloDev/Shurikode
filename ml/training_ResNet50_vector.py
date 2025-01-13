from model import Create_ResNet50_prob_vec
from dataset import shurikode_dataset_vector
from torch.optim import Adam
from tqdm import tqdm
from utils import save_model, acc_for_prob_vec
from typing import List

import torch
import argparse
import wandb
import os

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


m = Create_ResNet50_prob_vec().to(device)

epochs_n = 90
lr = 1e-4
train_variety = 400
val_variety = 30
batch_size = 40

optimizer = Adam(m.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
lambda_fn = lambda epoch: (
    0.5
    ** sum(
        epoch >= milestone
        for milestone in [0.4 * epochs_n, 0.6 * epochs_n, 0.8 * epochs_n]
    )
)

# Initialize LambdaLR
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)

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
train_dataset = shurikode_dataset_vector(
    data_path=args.datasets_path, type="train", variety=train_variety
)
train_dataloader = train_dataset.make_dataloader(batch_size=batch_size)
val_dataset = shurikode_dataset_vector(
    data_path=args.datasets_path, type="val", variety=val_variety
)
val_dataloader = val_dataset.make_dataloader(batch_size=batch_size)

loss_function = torch.nn.CrossEntropyLoss()

for epoch in range(epochs_n):

    ########################################## TRAINING EPOCH ##########################################
    m.train()
    tdataloader = tqdm(train_dataloader, unit="batch")
    for i, (img, gt) in enumerate(tdataloader):
        tdataloader.set_description(f"(training) Epoch {epoch}/{epochs_n}")
        img, gt = img.to(device), gt.to(device)

        pred: torch.Tensor = m(img)

        loss: torch.Tensor = loss_function(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tdataloader.set_postfix(
            loss=loss.item(),
        )

        acc = acc_for_prob_vec(device, pred, gt)

        tdataloader.set_postfix(loss=loss.item(), acc=acc)
        wandb.log({"loss": loss.item(), "acc_05": acc})

    ############################################ VALIDATION ###########################################
    tdataloader = tqdm(val_dataloader, unit="batch")
    loss_tower = []
    acc = 0
    with torch.no_grad():
        m.eval()
        for i, (img, gt) in enumerate(tdataloader):
            tdataloader.set_description(f"(validation) Epoch {epoch}/{epochs_n}")
            img, gt = img.to(device), gt.to(device)

            pred: torch.Tensor = m(img)

            acc = ((acc * i) + (acc_for_prob_vec(device, pred, gt) / batch_size)) / (
                i + 1
            )

            # loss: torch.Tensor = mse_loss_function(pred, gt)
            loss: torch.Tensor = loss_function(pred, gt)
            loss_tower.append(loss)

            tdataloader.set_postfix(loss=loss.item(), acc=acc)

        avg_loss = sum(loss_tower) / len(loss_tower)
        if is_first_epoch or avg_loss < min_loss:
            is_first_epoch = False
            min_loss = avg_loss

            checkpoint_filename = save_model(
                args.checkpoints_dir,
                f"e{epoch:03d}_{args.exp_name}",
                m.state_dict(),
            )
            checkpoint_files.append(checkpoint_filename)
            for ckpt_file in checkpoint_files[:-1]:
                if os.path.exists(ckpt_file):
                    os.remove(ckpt_file)
            checkpoint_files = checkpoint_files[-1:]
checkpoint_filename = save_model(
    args.checkpoints_dir,
    f"final_{args.exp_name}",
    m.state_dict(),
)
