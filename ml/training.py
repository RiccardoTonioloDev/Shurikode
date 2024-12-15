from model import CodeExtractor
from dataset import shurikode_dataset
from losses import loss_function
from torch.optim import Adam
from tqdm import tqdm
from utils import save_model
from typing import List

import torch
import argparse
import wandb
import numpy
import os

parser = argparse.ArgumentParser(
    description="Arguments for the training procedure of the Shurikode decoder model."
)
parser.add_argument("--exp_name", type=str, help="The name of the current experiment")
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    print("TRAINING ON CUDA")
elif torch.mps.is_available():
    device = "mps"
    print("TRAINING ON MPS")


m = CodeExtractor().to(device)

epochs_n = 90
lr = 1e-4
variety = 128
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
min_loss = numpy.zeros([0])
checkpoint_files: List[str] = []
for epoch in range(epochs_n):
    dataset = shurikode_dataset(variety=variety, epoch=epoch, epochs_n=epochs_n)

    dataloader = dataset.make_dataloader(batch_size=batch_size)

    epoch_loss = numpy.zeros([0])

    tdataloader = tqdm(dataloader, unit="batch")
    for img, gt in tdataloader:
        tdataloader.set_description(f"Epoch {epoch}/{epochs_n}")
        img, gt = img.to(device), gt.to(device)

        pred: torch.Tensor = m(img)

        loss: torch.Tensor = loss_function(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tdataloader.set_postfix(loss=loss.item())
        wandb.log({"loss": loss.item()})

        epoch_loss += loss.item()

    epoch_loss /= (256 * variety) / batch_size
    if is_first_epoch or epoch_loss < min_loss:
        is_first_epoch = False
        min_loss = epoch_loss

        checkpoint_filename = save_model(
            "./checkpoints/",
            f"e{epoch:03d}_{args.exp_name}",
            m.state_dict(),
        )
        checkpoint_files.append(checkpoint_filename)
        for ckpt_file in checkpoint_files[:-1]:
            if os.path.exists(ckpt_file):
                os.remove(ckpt_file)
        checkpoint_files = checkpoint_files[-1:]
