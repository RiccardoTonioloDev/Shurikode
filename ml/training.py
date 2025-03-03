from typing import Callable, Dict, List, Sequence
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor

from utils import (
    ModelEvaluationFunction,
    log_elapsed_remaining_total_time,
    ConsoleStatsLogger,
    ConditionalSave,
    Result,
)
from custom_types import DeviceType

import torch.nn as nn
import time
import torch
import wandb


def train(
    model: nn.Module,
    loss_function: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    clean_dataloader: DataLoader,
    device: DeviceType,
    evaluation_functions: Sequence[ModelEvaluationFunction],
    epoch_n: int,
    saver: ConditionalSave,
):
    console_logger = ConsoleStatsLogger(epoch_n)
    elapsed_time = 0

    for i in range(epoch_n):
        start_time_epoch = time.time()

        # Train the model on the train dataset
        train_epoch(
            model,
            loss_function,
            optimizer,
            train_dataloader,
            device,
            evaluation_functions,
        )

        # Validate the model on the validation dataset
        val_stats = validate_model(
            model, loss_function, val_dataloader, device, evaluation_functions
        )
        console_logger("Validation", val_stats, i)

        # Saving the model if it's performing better than the previously saved model
        saver(model, val_stats, i)

        # Validate the model on the clean (non augmented) dataset
        clean_stats = validate_model(
            model, loss_function, clean_dataloader, device, evaluation_functions
        )
        console_logger("Clean", clean_stats, i)

        # Logging on console time metrics
        end_time_epoch = time.time()
        elapsed_time += end_time_epoch - start_time_epoch
        log_elapsed_remaining_total_time(elapsed_time, i + 1, epoch_n)


def train_epoch(
    model: nn.Module,
    loss_function: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    dataloader: DataLoader,
    device: DeviceType,
    evaluation_functions: Sequence[ModelEvaluationFunction],
):
    """
    Given the model and the evaluation functions to be used, it trains the model on a single epoch, while loggin the
    various collected statistics in wandb.

    :param model: The model that will be trained.
    :param loss_function: The loss function to be used for training.
    :param optimizer: The optimizer that will be used for training.
    :param dataloader: The dataloader that will be used for training.
    :param device: The device that will be used to compute operations.
    :param evaluation_functions: A list of different evaluation functions that will be used to evaluate the model.
    """
    model.train()

    # Retrieving the batch sample & ground truth
    for img, gt in dataloader:
        img, gt = img.to(device), gt.to(device)

        # Calculating prediction and loss
        pred: Tensor = model(img)
        loss: Tensor = loss_function(pred, gt)

        # Optimizing
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating statistics and logging
        evals: Dict[str, float] = {}
        for eval_func in evaluation_functions:
            evals[eval_func.get_name()] = eval_func(pred, gt).get_value()
        wandb.log({"loss": loss.item(), **evals})


def validate_model(
    model: nn.Module,
    loss_function: Callable[[Tensor, Tensor], Tensor],
    dataloader: DataLoader,
    device: DeviceType,
    evaluation_functions: Sequence[ModelEvaluationFunction],
) -> List[Result]:
    """
    Given the model and the evaluation functions to be used, it will evaluate the model and return the collected
    statistics.

    :param model: The model that will be trained.
    :param loss_function: The loss function that was used for training
    :param dataloader: The dataloader that will be used for validation.
    :param device: The device that will be used to compute operations.
    :param evaluation_functions: A list of different evaluation functions that will be used to evaluate the model.

    :return: A list of the various evaluation statistics.
    """
    model.eval()

    evals = [0.0] * len(evaluation_functions)
    avg_loss = 0

    with torch.no_grad():
        for batch_idx, (img, gt) in enumerate(dataloader):
            img, gt = img.to(device), gt.to(device)

            pred: torch.Tensor = model(img)
            loss: torch.Tensor = loss_function(pred, gt)

            # Calculating dynamically the average statistics
            avg_loss = ((avg_loss * batch_idx) + loss.item()) / (batch_idx + 1)
            for func_idx, (eval_func) in enumerate(evaluation_functions):
                evals[func_idx] = (
                    (evals[func_idx] * batch_idx) + eval_func(pred, gt).get_value()
                ) / (batch_idx + 1)

    stats_w_names = [Result("Loss", avg_loss)] + [
        Result(evaluation_functions[i].get_name(), evals[i])
        for i in range(len(evaluation_functions))
    ]
    return stats_w_names
