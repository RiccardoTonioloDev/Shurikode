from custom_types import EvaluationType
from training import train, finetune
from model import Create_ResNet
from dataset import shurikode_dataset
from realset import real_dataset
from torch.optim import Adam
from torch.utils.data import ConcatDataset, random_split, DataLoader
from utils import (
    ConditionalSave,
    AverageAccuracyVectorOutput,
    find_device,
    set_seed,
    load_weights,
)

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
parser.add_argument(
    "--real_dataset_path",
    type=str,
    help="The directory where the real SHURIKODE dataset is located.",
)
parser.add_argument(
    "--selected_checkpoint_file_path",
    type=str,
    help="The path to the checkpoint file to be used.",
)
args = parser.parse_args()

# Model training parameters
N_CLASSES = 256
EPOCHS_N = 10
LR = 1e-4
TRAIN_VARIETY = 400
VAL_VARIETY = 30
CLEAN_VARIETY = 1
BATCH_SIZE = 32

# Checkpoint saving parameters
OBJECTIVE = "minimize"
METRIC: EvaluationType = "Loss"

device = find_device()

# MODEL CREATION #######################################################################################################
m = Create_ResNet("r18", n_classes=N_CLASSES, group_norm=True).to(device)
m.load_state_dict(load_weights(args.selected_checkpoint_file_path, device))

# OPTIMIZER & SCHEDULER CREATION #######################################################################################
optimizer = Adam(m.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
scheduler_function = lambda epoch: (
    0.5
    ** sum(
        epoch >= milestone
        for milestone in [0.4 * EPOCHS_N, 0.6 * EPOCHS_N, 0.8 * EPOCHS_N]
    )
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_function)


# LOSS & EVALUATION FUNCTIONS CREATION ###################################################################################
loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
evaluation_functions = [AverageAccuracyVectorOutput()]

# REAL DATASET CONFIGS
real_dtst = real_dataset(args.real_dataset_path)
set_seed(42)
real_train_dataset, real_val_dataset = random_split(real_dtst, [0.25, 0.75])
real_val_dataloader = DataLoader(real_val_dataset, BATCH_SIZE, False)

# DATASETS & DATALOADERS CREATION ######################################################################################
train_dataset = shurikode_dataset(
    data_path=args.datasets_path,
    type="train",
    variety=TRAIN_VARIETY,
    n_classes=N_CLASSES,
)
train_dataset = ConcatDataset([train_dataset, real_train_dataset])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_dataset = shurikode_dataset(
    data_path=args.datasets_path, type="val", variety=VAL_VARIETY, n_classes=N_CLASSES
)
val_dataloader = val_dataset.make_dataloader(batch_size=BATCH_SIZE)
clean_dataset = shurikode_dataset(
    data_path=args.datasets_path,
    type="clean",
    variety=CLEAN_VARIETY,
    n_classes=N_CLASSES,
)
clean_dataloader = clean_dataset.make_dataloader(batch_size=BATCH_SIZE)

wandb.init(
    project="Shurikode",
    name=args.exp_name,
    config={
        "num_epochs": EPOCHS_N,
        "learning_rate": LR,
    },
)

finetune(
    model=m,
    loss_function=loss_function,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    clean_dataloader=clean_dataloader,
    real_val_dataloader=real_val_dataloader,
    device=device,
    evaluation_functions=evaluation_functions,
    epoch_n=EPOCHS_N,
    saver=ConditionalSave(OBJECTIVE, METRIC, args.checkpoints_dir, args.exp_name),
)
