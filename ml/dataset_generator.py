from PIL.Image import Image
from shurikode.shurikode_encoder import shurikode_encoder
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from augmentators import (
    AugmentatorIf,
    RandomRotationDynamicFillerColor,
    RandomPerspectiveDynamicFillerColor,
    RandomScalerDynamicFillerColor,
    RandomWaveDistortion,
    RandomizeAugmentator,
)
from torchvision.transforms.v2 import Transform

import torchvision.transforms.v2 as t
import torch
import random
import os
import argparse


class shurikode_dataset_generator(Dataset):
    def __init__(
        self,
        transforms: Transform,
        variety: int = 100,
    ):
        """
        :param transforms: The basic PIL Image to tensor transformations plus optional augmentations.
        :param variety: The number of different representations for the same 2D encoded code.
        """
        self.__variety = variety
        self.__post_processing = transforms
        self.__encoder = shurikode_encoder(10)

    def __len__(self):
        return 256 * self.__variety

    def __getitem__(self, i: int) -> Tuple[Image, int]:
        value = i % 256
        image = self.__encoder.encode(value).get_PIL_image()
        image = self.__post_processing(image)
        return image, value

    def make_dataloader(
        self,
        batch_size: int = 1,
        shuffle_batch: bool = False,
        num_workers: int = 1,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        It creates a dataloader from the dataset.

        :param batch_size: The number of samples inside a single batch.
        :param shuffle_batch: If true the batches will be different in every epoch.
        :param num_workers: The number of workers used to create batches.
        :param pin_memory: Leave it to true (it's to optimize the flow of information between CPU and GPU).

        :return: The configured dataloader.
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
        "--clean_examples_dir",
        type=str,
        help="The directory that will be containing the not augmented validation images.",
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
    assert os.path.exists(
        args.clean_examples_dir
    ), f"The clean_examples_dir directory ({args.clean_examples_dir}) doesn't exist."

    IMAGES_DIAGONAL = 400

    post_processing_w_augmentations = t.Compose(
        [
            t.ToImage(),
            t.ToDtype(torch.float32, scale=True),
            t.Resize(IMAGES_DIAGONAL),
            AugmentatorIf(
                0.5,
                then=RandomRotationDynamicFillerColor((-90, 90), p=1),
                otherwise=RandomPerspectiveDynamicFillerColor(0.25, p=1),
            ),
            RandomScalerDynamicFillerColor(0, 0.05, diagonal=IMAGES_DIAGONAL, p=0.6),
            RandomWaveDistortion(IMAGES_DIAGONAL, IMAGES_DIAGONAL, 0.2, p=0.15),
            RandomizeAugmentator(t.GaussianBlur(9, 2.5), p=0.33),  # light
            RandomizeAugmentator(t.GaussianBlur(15, 4.75), p=0.33),  # strong
            RandomizeAugmentator(t.GaussianBlur(21, 7), p=0.33),  # stronger
            t.ToPILImage(),
        ]
    )
    post_processing_wo_augmentations = t.Compose(
        [
            t.ToImage(),
            t.ToDtype(torch.float32, scale=True),
            t.Resize(IMAGES_DIAGONAL),
            t.ToPILImage(),
        ]
    )

    dataloader = shurikode_dataset_generator(
        post_processing_w_augmentations,
        args.train_variety,
    ).make_dataloader()
    for idx, (pil_image, value) in enumerate(dataloader):
        series = int(idx / 256)
        pil_image.save(
            os.path.join(args.train_dir, f"{series:03}-{value.item():03}.png")
        )

    dataloader = shurikode_dataset_generator(
        post_processing_w_augmentations,
        args.val_variety,
    ).make_dataloader()
    for idx, (pil_image, value) in enumerate(dataloader):
        series = int(idx / 256)
        pil_image.save(os.path.join(args.val_dir, f"{series:03}-{value.item():03}.png"))

    dataloader = shurikode_dataset_generator(
        post_processing_wo_augmentations,
        1,
    ).make_dataloader()
    for idx, (pil_image, value) in enumerate(dataloader):
        series = int(idx / 256)
        pil_image.save(
            os.path.join(args.clean_examples_dir, f"{series:03}-{value.item():03}.png")
        )
