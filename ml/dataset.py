from torch.utils.data import Dataset, DataLoader
from utils import hamming_encode
from typing import List, Tuple, Literal
from torch import Tensor
from PIL import Image

import torchvision.transforms.v2 as transforms
import torch
import math
import os


class shurikode_dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        type: Literal["train", "val", "clean"],
        variety: int = 400,
        binary_output=False,
        hamming_bits=False,
        n_classes=256,
    ):
        """
        Creating the Shurikode dataset.

        :param data_path: The path where the `train/`, `val/` and `clean/` dataset directories are stored.
        :param type: The type of the dataset (basically what dataset subfolder is used in the class).
        :param variety: How many different examples of each class are there in the choosen dataset.
        :param binary_output: Wether the output is in a sparse vector or binary form.
        :param hamming_bits: Wether to add hamming correction bits to the output.
        :param n_classes: The number of different classes of the dataset.
        """
        assert (
            binary_output or not hamming_bits
        ), "You can't apply hamming correction on a model with a non-binary output."

        self.__original_n_classes = n_classes
        if binary_output:
            n_classes = math.ceil(math.log(n_classes, 2))
        self.__n_classes = n_classes
        self.__binary_output = binary_output
        self.__hamming_bits = hamming_bits
        self.__variety = variety
        self.__data_path = data_path
        self.__type = type
        self.__image_tensorizer = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        return self.__original_n_classes * self.__variety

    def __getitem__(self, i: int) -> Tuple[Tensor, int | List[bool]]:
        value = i % self.__original_n_classes
        series = int(i / self.__original_n_classes)

        image_path = os.path.join(
            self.__data_path, self.__type, f"{series:03}-{value:03}.png"
        )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor: torch.Tensor = self.__image_tensorizer(image)

        except Exception as e:
            raise RuntimeError(f"Error loading image: {image_path}. {e}")

        if self.__binary_output:
            value = self.__int_to_binary(value)
            if self.__hamming_bits:
                value = hamming_encode(value)

        return image_tensor, value

    def __int_to_binary(self, value: int) -> List[bool]:
        """
        Converts a value into a list of bool, representing a binary encoding of the initial value.

        :params value: The value to be encoded.
        """
        bit_array = [False] * self.__n_classes
        i = -1
        while value:  # Calculating the bit representation of the input value
            bit_array[i] = bool(1 & value)
            value = value >> 1
            i -= 1

        return bit_array

    def make_dataloader(
        self,
        batch_size: int = 8,
        shuffle_batch: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        It creates a dataloader from the dataset.

        :param batch_size: The number of samples inside a single batch;
        :param shuffle_batch: If true the batches will be different in every epoch;
        :param num_workers: The number of workers used to create batches;
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
