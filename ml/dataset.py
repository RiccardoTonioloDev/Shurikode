from typing import Tuple, Literal
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from PIL import Image

import torchvision.transforms.v2 as transforms
import torch
import os


class shurikode_dataset(Dataset):
    def __init__(
        self, data_path: str, type: Literal["train", "val"], variety: int = 400
    ):
        self.__variety = variety
        self.__data_path = data_path
        self.__type = type
        self.__image_tensorizer = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=False),
            ]
        )

    def __len__(self):

        return 256 * self.__variety

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        value = i % 256
        series = int(i / 256)

        image_path = os.path.join(
            self.__data_path, self.__type, f"{series:03}-{value:03}.png"
        )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor: torch.Tensor = self.__image_tensorizer(image) / 255

        except Exception as e:
            raise RuntimeError(f"Error loading image: {image_path}. {e}")

        # multiclass_vector = torch.zeros([256])
        # multiclass_vector[value] = 1

        bit_tensor = torch.zeros([8])
        idx = -1
        while value:
            bit_tensor[idx] = 1 & value
            value = value >> 1
            idx -= 1

        return image_tensor.clamp(0, 1), bit_tensor  # , multiclass_vector

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
