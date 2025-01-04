from PIL import Image
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
import os
import argparse
import PIL


class shurikode_dataset_generator(Dataset):
    def __init__(self, variety: int = 100):
        self.__variety = variety

        self.__image_tensorizer = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )

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

    def __getitem__(self, i: int) -> Tuple[Tensor, int]:
        value = i % 256

        code_tensor: Tensor = self.__image_tensorizer(
            self.__shurikode_encoder.encode(value).get_PIL_image()
        ).unsqueeze(0)

        code_tensor = torch.nn.functional.interpolate(
            code_tensor, (400, 400), mode="bilinear"
        )

        aug_choice = random.random()
        if aug_choice < 0.2375:
            code_tensor: Tensor = self.__clear_complete_augs(code_tensor)
        elif aug_choice < 0.475:
            code_tensor: Tensor = self.__distorted_complete_augs(code_tensor)
        elif aug_choice < 0.7125:
            code_tensor: Tensor = self.__clear_piece_augs(code_tensor)
        elif aug_choice < 0.95:
            code_tensor: Tensor = self.__distorted_piece_augs(code_tensor)

        return code_tensor.squeeze(0), value

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


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    parser = argparse.ArgumentParser(
        description="Arguments for the dataset creation procedure of the Shurikode decoder model."
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        help="The directory that will be containing the training images.",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        help="The directory that will be containing the validation images.",
    )
    parser.add_argument(
        "--train_variety",
        type=int,
        help="The variety of each class in the training dataset.",
        default=400,
    )
    parser.add_argument(
        "--val_variety",
        type=int,
        help="The variety of each class in the validation dataset.",
        default=30,
    )
    args = parser.parse_args()
    assert os.path.exists(
        args.train_dir
    ), f"The train_dir directory ({args.train_dir}) doesn't exist."
    assert os.path.exists(
        args.val_dir
    ), f"The val_dir directory ({args.val_dir}) doesn't exist."
    to_pil_image = transforms.ToPILImage()
    dataloader = shurikode_dataset_generator(
        args.train_variety,
    ).make_dataloader(1, False)
    for idx, (img, value) in enumerate(dataloader):
        pil_image: Image.Image = to_pil_image(torch.clamp(img[0], 0, 255))
        series = int(idx / 256)
        pil_image.save(
            os.path.join(args.train_dir, f"{series:03}-{value.item():03}.png")
        )
    dataloader = shurikode_dataset_generator(
        args.val_variety,
    ).make_dataloader(1, False)
    for idx, (img, value) in enumerate(dataloader):
        pil_image: Image.Image = to_pil_image(torch.clamp(img[0], 0, 255))
        series = int(idx / 256)
        pil_image.save(os.path.join(args.val_dir, f"{series:03}-{value.item():03}.png"))
