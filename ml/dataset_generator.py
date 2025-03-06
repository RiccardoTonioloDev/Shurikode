from PIL.Image import Image
from shurikode.shurikode_encoder import shurikode_encoder
from typing import Tuple
from augmentators import (
    AugmentatorIf,
    RandomBrightnessAdjust,
    RandomPieceCutterDynamicFillerColor,
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


class shurikode_dataset_generator:
    def __init__(self, transforms: Transform, variety: int = 100, num_classes=256):
        """
        :param transforms: The basic PIL Image to tensor transformations plus optional augmentations.
        :param variety: The number of different representations for the same 2D encoded code.
        """
        self.__variety = variety
        self.__post_processing = transforms
        self.__encoder = shurikode_encoder(10)
        self.__num_classes = num_classes

    def __len__(self):
        return self.__num_classes * self.__variety

    def __getitem__(self, i: int) -> Tuple[Image, int]:
        value = i % self.__num_classes
        image = self.__encoder.encode(value).get_PIL_image()
        image = self.__post_processing(image)
        return image, value


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
                then=RandomRotationDynamicFillerColor((-90, 90), p=0.7),
                otherwise=RandomPerspectiveDynamicFillerColor(0.25, p=0.7),
            ),
            RandomPieceCutterDynamicFillerColor(IMAGES_DIAGONAL, p=0.1),
            RandomScalerDynamicFillerColor(0.0, 0.20, diagonal=IMAGES_DIAGONAL, p=0.6),
            RandomWaveDistortion(IMAGES_DIAGONAL, IMAGES_DIAGONAL, 0.2, p=0.15),
            RandomizeAugmentator(t.GaussianBlur(9, 2.5), p=0.33),  # light
            RandomizeAugmentator(t.GaussianBlur(15, 4.75), p=0.33),  # strong
            RandomizeAugmentator(t.GaussianBlur(21, 7), p=0.33),  # stronger
            RandomizeAugmentator(t.GaussianNoise(0, 0.05), p=0.3),  # stronger
            RandomizeAugmentator(t.GaussianNoise(0, 0.1), p=0.2),  # stronger
            RandomizeAugmentator(t.GaussianNoise(0, 0.2), p=0.1),  # stronger
            RandomBrightnessAdjust(p=0.5),
            t.RandomErasing(p=0.4, scale=(0.1, 0.3)),
            t.RandomErasing(p=0.4, scale=(0.1, 0.3)),
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

    NUM_CLASSES = 256

    dataset = shurikode_dataset_generator(
        post_processing_w_augmentations, args.train_variety, NUM_CLASSES
    )
    for i in range(NUM_CLASSES * args.train_variety):
        series = int(i / 256)
        pil_image, value = dataset[i]
        pil_image.save(os.path.join(args.train_dir, f"{series:03}-{value:03}.png"))

    dataset = shurikode_dataset_generator(
        post_processing_w_augmentations, args.val_variety, NUM_CLASSES
    )
    for i in range(NUM_CLASSES * args.val_variety):
        series = int(i / 256)
        pil_image, value = dataset[i]
        pil_image.save(os.path.join(args.val_dir, f"{series:03}-{value:03}.png"))

    dataset = shurikode_dataset_generator(
        post_processing_wo_augmentations, 1, NUM_CLASSES
    )
    for i in range(NUM_CLASSES):
        series = int(i / 256)
        pil_image, value = dataset[i]
        pil_image.save(
            os.path.join(args.clean_examples_dir, f"{series:03}-{value:03}.png")
        )
