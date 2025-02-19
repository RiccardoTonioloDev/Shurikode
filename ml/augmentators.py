from torchvision.transforms.v2 import functional as F, InterpolationMode
import torchvision.transforms.v2 as transforms
from typing import List, Tuple

import torch
import random


class RandomRotationWithColor:
    def __init__(self, degrees, expand=True):
        self.degrees = degrees
        self.expand = expand

    def __call__(self, img: torch.Tensor, filler_color: List[float]):
        assert len(filler_color) == 3, "filler_color must be of length 3."
        # Genera colore casuale
        return F.rotate(
            img,
            angle=random.uniform(*self.degrees),
            expand=self.expand,
            fill=filler_color,
            interpolation=InterpolationMode.NEAREST,
        )


class RandomPerspectiveWithColor:
    def __init__(
        self, distortion_scale=0.5, p=0.5, interpolation=F.InterpolationMode.NEAREST
    ):
        self.__distortion_scale = distortion_scale
        self.__p = p
        self.__interpolation = interpolation

    def __call__(self, img: torch.Tensor, filler_color: List[float]):
        return transforms.RandomPerspective(
            p=self.__p,
            distortion_scale=self.__distortion_scale,
            interpolation=self.__interpolation,
            fill=filler_color,
        )(img)


class RandomRotationPerspectiveWithColor:
    def __init__(
        self,
        degrees_rotation=[-90, 90],
        p_perspective=0.5,
        distortion_scale=0.5,
        diagonal=400,
    ):
        self.__rotation = RandomRotationWithColor(degrees_rotation, True)
        self.__perspective = RandomPerspectiveWithColor(distortion_scale, p_perspective)
        self.__diagonal = diagonal

    def __call__(self, img: torch.Tensor):
        filler_color = [random.random(), random.random(), random.random()]
        # ----------- THE PROBLEM IS NOT HERE
        img = self.__rotation(img, filler_color)
        img = self.__perspective(img, filler_color)
        img = img.unsqueeze(0)
        return torch.nn.functional.interpolate(
            img, (self.__diagonal, self.__diagonal), mode="bilinear"
        )


class PieceCutter:
    def __init__(self, diagonal=400):
        self.__diagonal = diagonal
        pass

    def __call__(self, img: torch.Tensor, filler_color: List[float]):
        portion_size_percentage = 0.4 + 0.2 * random.random()
        portion_size = int(img.shape[-1] * portion_size_percentage)
        is_side = random.random() > 0.5
        if not is_side:
            r = random.uniform(-90, 90)
            img = F.rotate(img, r, expand=True, fill=filler_color)
        which_side = random.random()
        if which_side < 0.25:  # TOP
            img = img[:, :, :portion_size, :]
        elif which_side < 0.50:  # BOTTOM
            img = img[:, :, img.shape[-1] - portion_size :, :]
        elif which_side < 0.75:  # LEFT
            img = img[:, :, :, :portion_size]
        else:  # RIGHT
            img = img[:, :, :, img.shape[-1] - portion_size :]
        return torch.nn.functional.interpolate(
            img, (self.__diagonal, self.__diagonal), mode="bilinear"
        )


class RandomPerspectivePieceCutter:
    def __init__(self, p_perspective=0.5, distortion_scale=0.5, diagonal=400):
        self.__piece = PieceCutter(diagonal)
        self.__perspective = RandomPerspectiveWithColor(distortion_scale, p_perspective)

    def __call__(self, img: torch.Tensor):
        filler_color = [random.random(), random.random(), random.random()]
        assert (
            len(img.shape) == 3
        ), "The shape of the input tensor for this module must be 3D."
        img = img.unsqueeze(0)
        img = self.__piece(img, filler_color)
        return self.__perspective(img, filler_color)


class RandomScaler:
    def __init__(self, min_pad=0.0, max_pad=0.3, diagonal=400):
        self.__min_pad = min_pad * diagonal
        self.__max_pad = max_pad * diagonal
        self.__diagonal = diagonal

    def __call__(self, img: torch.Tensor, color: List[float]):
        r = random.random()
        pad = int((self.__max_pad - self.__min_pad) * r + self.__min_pad)
        padder = transforms.Pad(padding=pad, fill=color, padding_mode="constant")
        img = padder(img)
        return img


class RandomScaleRotationPerspectiveWithColor:
    def __init__(
        self,
        min_pad=0.0,
        max_pad=0.3,
        degrees_rotation=[-90, 90],
        p_perspective=0.5,
        distortion_scale=0.5,
        diagonal=400,
    ):
        self.__scaler = RandomScaler(min_pad, max_pad, diagonal)
        self.__rotation = RandomRotationWithColor(degrees_rotation, True)
        self.__perspective = RandomPerspectiveWithColor(distortion_scale, p_perspective)
        self.__diagonal = diagonal

    def __call__(self, img: torch.Tensor):
        filler_color = [random.random(), random.random(), random.random()]
        img = self.__scaler(img, filler_color)
        img = self.__rotation(img, filler_color)
        img = self.__perspective(img, filler_color)
        img = img.unsqueeze(0)
        return torch.nn.functional.interpolate(
            img, (self.__diagonal, self.__diagonal), mode="bilinear"
        )
