from shurikode.ml_model.m import Create_ResNet50
from typing import Tuple, Union, cast
from PIL.Image import Image
import torchvision.transforms.v2 as transforms

import torch
import torchvision


class shurikode_decoder:
    def __init__(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        self.__device = device
        self.__m = Create_ResNet50(device).to(device)
        self.__image_tensorizer = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=False),
            ]
        )

    def __call__(self, img: Union[Image, torch.Tensor]) -> int:
        if isinstance(img, Image):
            img = self.__image_tensorizer(img)
        if isinstance(img, torch.Tensor):
            if len(img.shape) < 4:
                img = img.unsqueeze(0)
        img = torch.nn.functional.interpolate(img, (400, 400), mode="bilinear")
        img = cast(torch.Tensor, img).to(self.__device)

        out: torch.Tensor = torch.softmax(self.__m(img), 1).squeeze(0)

        return out.argmax(-1).item().__int__()
