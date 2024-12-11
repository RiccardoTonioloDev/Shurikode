from shurikode.shurikode_encoder import shurikode_encoder
from typing import Tuple
from torch.utils.data import Dataset
from torch import Tensor

import torchvision.transforms.v2 as transforms
import torch
import random


class shurikode_training_dataset(Dataset):
    def __init__(self, variety: int = 100):
        self.__variety = variety

        self.__image_tensorizer = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )

        # self.__bad_augmentators = transforms.Compose(
        #     [
        #         transforms.RandomRotation([-90, 90], expand=True),
        #         transforms.RandomPerspective(0.3),
        #         transforms.ElasticTransform(50, 1),
        #         transforms.RandomErasing(0.33),
        #         transforms.ColorJitter(0.8),
        #         transforms.GaussianBlur(9),
        #     ]
        # )
        # self.__good_augmentators = transforms.Compose(
        #     [
        #         transforms.RandomRotation([-90, 90], expand=True),
        #         transforms.RandomPerspective(0.3),
        #         transforms.RandomErasing(0.4, (0.1, 0.4)),
        #         transforms.ColorJitter(0.8),
        #         transforms.GaussianBlur(9),
        #         transforms.GaussianNoise(sigma=0.5),
        #     ]
        # )

        self.__better_augmentators = transforms.Compose(
            [
                transforms.RandomRotation([-90, 90], expand=True),
                transforms.RandomPerspective(0.3, 1),
                transforms.RandomErasing(0.3, (0.1, 0.4)),
                transforms.RandomErasing(0.3, (0.1, 0.4)),
                transforms.GaussianBlur(31, 20),
            ]
        )

        self.__shurikode_encoder = shurikode_encoder(10)

    def __len__(self):

        return 256 * self.__variety

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        i = i % 256

        code_tensor: Tensor = self.__image_tensorizer(
            self.__shurikode_encoder.encode(i).get_PIL_image()
        ).unsqueeze(0)

        code_tensor = torch.nn.functional.interpolate(
            code_tensor, (400, 400), mode="bilinear"
        )

        code_tensor: Tensor = self.__better_augmentators(code_tensor)

        label = torch.zeros([256])
        label[i] = 1

        return code_tensor.clamp(0, 1), label

    # - Random rotation
    # - Random perspective
    # - Elastic transform
    # - Color jitter
    # - Gaussian Blur
    # - Gaussian Noise
    # - Random erasing
