from typing import Dict, Any, Literal, List
from abc import ABC, abstractmethod
from torch import Tensor

import torch.nn as nn
import torch
import os

from custom_types import EvaluationType


def find_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("TRAINING ON CUDA")
    elif torch.mps.is_available():
        device = "mps"
        print("TRAINING ON MPS")
    return device


def save_model(
    checkpoints_path: str, name: str, model_state_dict: Dict[str, Any]
) -> str:
    checkpoint_filename = os.path.join(checkpoints_path, f"checkpoint_{name}.pth.tar")
    torch.save(model_state_dict, checkpoint_filename)
    return checkpoint_filename


class Metric:
    def __init__(self, name: str, val: float) -> None:
        self.__name = name
        self.__val = val

    def get_name(self):
        return self.__name

    def get_value(self):
        return self.__val


class ModelEvaluationFunction(ABC):
    def __init__(self, name: EvaluationType) -> None:
        super().__init__()
        self.__name = name

    def get_name(self):
        return self.__name

    @abstractmethod
    def __call__(self, pred: Tensor, gt: Tensor) -> Metric:
        pass


class AccuracyVectorOutput(ModelEvaluationFunction):
    def __init__(self) -> None:
        super().__init__("Accuracy")

    def __call__(self, pred: Tensor, gt: Tensor) -> Metric:
        batch_size = pred.shape[0]
        pred = torch.softmax(pred, dim=1).argmax(dim=1)
        corrects = (pred == gt).int().sum()
        return Metric(self.get_name(), corrects.item() / batch_size)


class ConsoleStatsLogger:
    def __init__(self, epoch_n: int) -> None:
        self.__epoch_n = epoch_n

    def __call__(self, type: str, stats: List[Metric], epoch: int) -> Any:
        str_stats = "| "
        for stat in stats:
            str_stats = str_stats + f"{stat.get_name()}: {stat.get_value():.4f} | "
        print(f"({type}) Epoch {epoch}/{self.__epoch_n}: {str_stats}")


def log_elapsed_remaining_total_time(
    elapsed_time: float, epochs_done: int, epochs_n: int
) -> None:
    e_hours, remainder = divmod(elapsed_time, 3600)
    e_minutes, e_seconds = divmod(remainder, 60)

    total_time = elapsed_time / epochs_done * epochs_n
    t_hours, remainder = divmod(total_time, 3600)
    t_minutes, t_seconds = divmod(remainder, 60)

    remaining_time = total_time - elapsed_time
    r_hours, remainder = divmod(remaining_time, 3600)
    r_minutes, r_seconds = divmod(remainder, 60)

    print(f"ELAPSED TIME: {e_hours}h {e_minutes}m {e_seconds:.2f}s")
    print(f"REMAINING TIME: {r_hours}h {r_minutes}m {r_seconds:.2f}s")
    print(f"TOTAL TIME: {t_hours}h {t_minutes}m {t_seconds:.2f}s")


class ConditionalSave:
    def __init__(
        self,
        objective: Literal["minimize", "maximize"],
        metric_name: str,
        checkpoints_dir: str,
        experiment_name: str,
    ) -> None:
        self.__checkpoint_files: List[str] = []
        self.__best_evaluation = float("-inf")
        if objective == "minimize":
            self.__best_evaluation = float("inf")
        self.__metric_name = metric_name
        self.__objective = objective
        self.__checkpoints_dir = checkpoints_dir
        self.__experiment_name = experiment_name

    def __call__(
        self,
        model: nn.Module,
        model_stats: List[Metric],
        epoch: int,
    ) -> None:
        # Finding the evaluation metric index from the various statistics retrieved
        metric_index = [
            i
            for i, metric in enumerate(model_stats)
            if metric.get_name() == self.__metric_name
        ]
        assert (
            len(metric_index) != 0
        ), f"Metric name {self.__metric_name} for conditional saving doesn't exist within defined metrics."
        current_evaluation = model_stats[metric_index[0]].get_value()

        # Guard for correct objective prediction
        is_greater_than_best = current_evaluation > self.__best_evaluation
        is_maximising = self.__objective == "maximize"
        if (not (is_greater_than_best != is_maximising)) or abs(
            self.__best_evaluation
        ) == float("inf"):

            self.__best_evaluation = current_evaluation
            checkpoint_filename = save_model(
                self.__checkpoints_dir,
                f"e{epoch:03d}_{self.__experiment_name}",
                model.state_dict(),
            )
            self.__checkpoint_files.append(checkpoint_filename)
            for ckpt_file in self.__checkpoint_files[:-1]:
                if os.path.exists(ckpt_file):
                    os.remove(ckpt_file)
            self.__checkpoint_files = self.__checkpoint_files[-1:]
