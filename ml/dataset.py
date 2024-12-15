from shurikode.shurikode_encoder import shurikode_encoder
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from augmentators import (
    RandomRotationPerspectiveWithColor,
    RandomPerspectivePieceCutter,
)

import torchvision.transforms.v2 as transforms
import torch
import random


class shurikode_dataset(Dataset):
    def __init__(self, variety: int = 100, epoch: int = 0, epochs_n: int = 100):
        self.__variety = variety

        self.__image_tensorizer = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )

        self.__progress = epoch / epochs_n

        filler_color = (
            random.random(),
            random.random(),
            random.random(),
        )

        # The code it's clear, there are little perspective changes, rotation and 2 random erasing (40% -> 25%)
        self.__clear_complete_augs = transforms.Compose(
            [
                RandomRotationPerspectiveWithColor([-90, 90], 1, 0.2, 400),
                transforms.RandomErasing(0.3, (0.1, 0.4)),
                transforms.RandomErasing(0.3, (0.1, 0.4)),
                transforms.GaussianBlur(7, (1.5, 2.5)),
            ]
        )

        # The code it's blurred, there are little perspective changes, rotation, gaussian noise, and 1 random erasing (20% -> 25%)
        self.__distorted_complete_augs = transforms.Compose(
            [
                RandomRotationPerspectiveWithColor([-90, 90], 1, 0.2, 400),
                transforms.RandomErasing(0.3, (0.1, 0.4)),
                transforms.GaussianBlur(25, 10),
                transforms.GaussianNoise(sigma=0.1),
            ]
        )

        # Only a portion of the code is visible, but it's clear, there are little perspective changes (20% -> 25%)
        self.__clear_piece_augs = transforms.Compose(
            [
                RandomPerspectivePieceCutter(1, 0.3, 400),
                transforms.GaussianBlur(7, (1.5, 2.5)),
            ]
        )

        # Only a portion of the code is visible, but it's blurred, there are little perspective changes, gaussian noise (20% -> 25%)
        self.__distorted_piece_augs = transforms.Compose(
            [
                RandomPerspectivePieceCutter(1, 0.3, 400),
                transforms.GaussianBlur(25, 10),
                transforms.GaussianNoise(sigma=0.1),
            ]
        )

        self.__shurikode_encoder = shurikode_encoder(10)

    def __len__(self):

        return 256 * self.__variety

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        value = i % 256

        code_tensor: Tensor = self.__image_tensorizer(
            self.__shurikode_encoder.encode(value).get_PIL_image()
        ).unsqueeze(0)

        code_tensor = torch.nn.functional.interpolate(
            code_tensor, (400, 400), mode="bilinear"
        )

        aug_choice = random.random()
        if aug_choice < 0.4 - 0.15 * self.__progress:
            code_tensor: Tensor = self.__clear_complete_augs(code_tensor)
        elif aug_choice < 0.6 - 0.10 * self.__progress:
            code_tensor: Tensor = self.__distorted_complete_augs(code_tensor)
        elif aug_choice < 0.8 - 0.05 * self.__progress:
            code_tensor: Tensor = self.__clear_piece_augs(code_tensor)
        else:
            code_tensor: Tensor = self.__distorted_piece_augs(code_tensor)

        bit_tensor = torch.zeros([8])
        idx = -1
        while value:
            bit_tensor[idx] = 1 & value
            value = value >> 1
            idx -= 1

        return code_tensor.clamp(0, 1).squeeze(0), bit_tensor

    def make_dataloader(
        self,
        batch_size: int = 8,
        shuffle_batch: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
            It creates a dataloader from the dataset.
            - `batch_size`: the number of samples inside a single batch;
            - `shuffle_batch`: if true the batches will be different in every epoch;
            - `num_workers`: the number of workers used to create batches;
            - `pin_memory`: leave it to true (it's to optimize the flow of information between CPU and GPU).

        Returns the configured dataloader.
        """

        dataloader = DataLoader(
            self,
            batch_size,
            shuffle_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return dataloader
