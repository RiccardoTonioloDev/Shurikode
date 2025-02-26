from PIL import Image
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

import torchvision.transforms.v2 as t
import torch
import random
import os
import argparse


class shurikode_dataset_generator(Dataset):
    """
    - 50% clear
        - 50% random rotation, random perspective
        - 50% gaussian blur, random rotation, random perspective
    - 50% distorted
        - 50% with: piece, gaussian blur, random rotation, random perspective
        - 50% with: gaussian blur, 1 random erasing, random perspective, random rotation
    """

    def __init__(self, variety: int = 100, diagonal: int = 400):
        self.__variety = variety
        self.__diagonal = diagonal

        self.__image_tensorizer = t.Compose(
            [
                t.ToImage(),
                t.ToDtype(torch.float32, scale=True),
                t.Resize(self.__diagonal),
            ]
        )

        self.__post_processing = t.Compose(
            [
                AugmentatorIf(
                    0.5,
                    then=RandomRotationDynamicFillerColor(1, (-90, 90)),
                    otherwise=RandomPerspectiveDynamicFillerColor(1, 0.25),
                ),
                RandomScalerDynamicFillerColor(1.0, 0, 0.05),
                RandomWaveDistortion(1, 400, 400, 0.2),
                RandomizeAugmentator(t.GaussianBlur(9, 2.5), 0.33),  # light
                RandomizeAugmentator(t.GaussianBlur(15, 4.75), 0.33),  # strong
                RandomizeAugmentator(t.GaussianBlur(21, 7), 0.33),  # strong
            ]
        )

        self.__encoder = shurikode_encoder(10)

    def __len__(self):
        return 256 * self.__variety

    def __getitem__(self, i: int) -> Tuple[Tensor, int]:
        value = i % 256

        code_image = self.__encoder.encode(value).get_PIL_image()
        code_tensor: Tensor = self.__image_tensorizer(code_image)

        code_tensor = self.__post_processing(code_tensor)

        return code_tensor, value

    def make_dataloader(
        self,
        batch_size: int = 1,
        shuffle_batch: bool = False,
        num_workers: int = 1,
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

    to_pil_image = t.ToPILImage()

    dataloader = shurikode_dataset_generator(
        args.train_variety,
    ).make_dataloader()
    for idx, (img, value) in enumerate(dataloader):
        pil_image: Image.Image = to_pil_image(torch.clamp(img[0], 0, 255))
        series = int(idx / 256)
        pil_image.save(
            os.path.join(args.train_dir, f"{series:03}-{value.item():03}.png")
        )

    dataloader = shurikode_dataset_generator(
        args.val_variety,
    ).make_dataloader()
    for idx, (img, value) in enumerate(dataloader):
        pil_image: Image.Image = to_pil_image(torch.clamp(img[0], 0, 255))
        series = int(idx / 256)
        pil_image.save(os.path.join(args.val_dir, f"{series:03}-{value.item():03}.png"))
