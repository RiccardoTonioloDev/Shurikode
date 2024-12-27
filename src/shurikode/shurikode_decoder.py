from shurikode.ml_model.m import Create_ResNet50
from typing import Tuple, Union, cast
from PIL.Image import Image
import torchvision.transforms.v2 as transforms

import torch


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

    def __call__(
        self, img: Union[Image, torch.Tensor]
    ) -> Tuple[int, int, int, int, int, int, int, int]:
        if isinstance(img, Image):
            img = self.__image_tensorizer(img)
        if isinstance(img, torch.Tensor):
            if len(img.shape) < 4:
                img = img.unsqueeze(0)
        img = torch.nn.functional.interpolate(img, (400, 400), mode="bilinear")
        img = cast(torch.Tensor, img).to(self.__device)
        img = img / 255

        out: torch.Tensor = self.__m(img).squeeze(0)

        out = (out > 0.5) * torch.ones(out.shape).to(self.__device)

        return (
            out[0].item().__int__(),
            out[1].item().__int__(),
            out[2].item().__int__(),
            out[3].item().__int__(),
            out[4].item().__int__(),
            out[5].item().__int__(),
            out[6].item().__int__(),
            out[7].item().__int__(),
            # out[9].item().__int__(),
            # out[10].item().__int__(),
            # out[11].item().__int__(),
        )
