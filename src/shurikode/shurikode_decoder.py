from shurikode.ml_model.m import Create_ResNet_Shurikode
from typing import Union, cast
from PIL.Image import Image
from torch import Tensor
from shurikode.utils import find_device, ModelType
import torchvision.transforms.v2 as transforms

import torch


class ShurikodeDecoder:
    __image_tensorizer = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    def __init__(self, m: ModelType = "r50"):
        """
        Initializes a `shurikode_decoder` object.

        :param `m`: The model to be used in the decoder to decode the images. The possible models are 'r18', 'r34' and
        'r50'.
        """
        self.__device = find_device()
        self.__m = Create_ResNet_Shurikode(
            m, 256, self.__device, group_norm=True
        ).eval()

    def __call__(self, img: Union[Image, Tensor]) -> int:
        """
        Given a single Pillow `Image` or a `torch.Tensor` representing just one image (both as a 3D tensor or as a 4D
        tensor of batch size 1), returns the shurikode label of that specifc image (assuming that the image contains
        a shurikode encoded code).

        :param `img`: The Pillow Image or `torch.Tensor` (as a 3D tensor or a 4D tensor of batch size 1).
        """
        with torch.no_grad():
            img_t = self.__img_to_expected_tensor(img, self.__device)
            model_output_logits: Tensor = self.__m(img_t).squeeze(0)
            label = int(model_output_logits.argmax(-1).item())
        return label

    @staticmethod
    def __img_to_expected_tensor(img: Union[Image, Tensor], device: str) -> Tensor:
        """
        Given a single Pillow `Image` or a `torch.Tensor` representing just one image (both as a 3D tensor or as a 4D
        tensor of batch size 1), returns a 4D tensor representing the image in the desired size and format of the
        selected decoder model.

        :param `img`: The Pillow Image or `torch.Tensor` (as a 3D tensor or a 4D tensor of batch size 1).
        :param `device`: The device on which to load the tensors (and the model). Possible choices are "cpu", "cuda" and
        "mps".
        """
        assert device in [
            "cpu",
            "cuda",
            "mps",
        ], f"The device type has to be between 'cuda', 'cpu' and 'mps'. Provided device: {device}."
        if isinstance(img, Image):
            img = cast(Tensor, ShurikodeDecoder.__image_tensorizer(img))
        if isinstance(img, Tensor):
            if len(img.shape) < 4:
                img = img.unsqueeze(0)
        assert (
            img.shape[0] == 1
        ), f"The encoder accepts only one image at a time. {img.shape[0]} images provided."

        img = torch.nn.functional.interpolate(img, (400, 400), mode="bilinear")
        img = cast(
            Tensor, img
        )  # Removing type ambiguity due to the interpolate function

        return img.to(device)
