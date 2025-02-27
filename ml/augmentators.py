from abc import ABC, abstractmethod
from torchvision.transforms.v2 import functional as F, InterpolationMode
import torchvision.transforms.v2 as transforms
from typing import Any, Callable, List, Sequence, Tuple
from random import random
from torch import Tensor

import torch
import random


class ColorGenerator:
    def __init__(self) -> None:
        self.__i = 0
        self.__fill = (random.random(), random.random(), random.random())
        self.__finished_subscribing = False
        self.__subscribers: List[object] = []

    def subscribe(self, subscriber: object) -> None:
        assert (
            not self.__finished_subscribing
        ), "After you called get_color for the first time, you can no longer \
            subscribe with other augmentators."
        self.__subscribers.append(subscriber)

    def get_color(self, subscriber: object) -> Tuple[float, float, float]:
        self.__finished_subscribing = True
        sub_idx = self.__subscribers.index(subscriber)
        if self.__i < sub_idx:
            self.__i = sub_idx
        elif self.__i > sub_idx:
            self.__fill = (random.random(), random.random(), random.random())
            self.__i = sub_idx
        self.__i += 1
        return self.__fill


ColorType = Tuple[float, float, float]

color_gen = ColorGenerator()


class DynamicColorAugmentators(ABC):

    def __init__(self) -> None:
        self.__color_gen = color_gen
        self.__color_gen.subscribe(self)

    def __call__(self, img: Tensor) -> Tensor:
        fill = self.__color_gen.get_color(self)
        return self._augment(img, fill)

    @abstractmethod
    def _augment(self, img: Tensor, fill: ColorType) -> Tensor:
        pass


class RandomRotationDynamicFillerColor(DynamicColorAugmentators):
    def __init__(
        self, degrees: Tuple[int, int], diagonal: int = 400, expand=True, p: float = 0.5
    ):
        super().__init__()
        self.__degrees = degrees
        self.__expand = expand
        self.__p = p
        self.__diagonal = diagonal

    def _augment(self, img: Tensor, fill: ColorType) -> Tensor:
        if random.random() > self.__p:
            return img
        assert len(fill) == 3, "filler_color must be of length 3."
        assert len(img.shape) == 3, "The provided image must only have 3 dimensions."
        angle = random.uniform(*self.__degrees)
        rotated_img = F.rotate(
            img,
            angle=angle,
            expand=self.__expand,
            fill=[fill[0], fill[1], fill[2]],
            interpolation=InterpolationMode.NEAREST,
        )
        return torch.nn.functional.interpolate(
            rotated_img.unsqueeze(0),
            (self.__diagonal, self.__diagonal),
            mode="bilinear",
        ).squeeze(0)


class RandomPerspectiveDynamicFillerColor(DynamicColorAugmentators):
    def __init__(
        self,
        distortion_scale=0.5,
        interpolation=F.InterpolationMode.NEAREST,
        p: float = 0.5,
    ):
        super().__init__()
        self.__distortion_scale = distortion_scale
        self.__p = p
        self.__interpolation = interpolation

    def _augment(self, img: Tensor, fill: ColorType) -> torch.Tensor:
        if random.random() > self.__p:
            return img
        startpoints, endpoints = self.__compute_start_end_points(
            img.shape[-2], img.shape[-1]
        )
        return F.perspective(
            img,
            interpolation=self.__interpolation,
            fill=[fill[0], fill[1], fill[2]],
            startpoints=startpoints,
            endpoints=endpoints,
        )

    def __compute_start_end_points(self, height: int, width: int):
        height, width = height, width

        distortion_scale = self.__distortion_scale

        half_height = height // 2
        half_width = width // 2
        bound_height = int(distortion_scale * half_height) + 1
        bound_width = int(distortion_scale * half_width) + 1
        topleft = [
            int(torch.randint(0, bound_width, size=(1,))),
            int(torch.randint(0, bound_height, size=(1,))),
        ]
        topright = [
            int(torch.randint(width - bound_width, width, size=(1,))),
            int(torch.randint(0, bound_height, size=(1,))),
        ]
        botright = [
            int(torch.randint(width - bound_width, width, size=(1,))),
            int(torch.randint(height - bound_height, height, size=(1,))),
        ]
        botleft = [
            int(torch.randint(0, bound_width, size=(1,))),
            int(torch.randint(height - bound_height, height, size=(1,))),
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]

        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints


class RandomPieceCutterDynamicFillerColor(DynamicColorAugmentators):
    def __init__(
        self,
        diagonal=400,
        portion_interval: Tuple[float, float] = (0.5, 1.0),
        p: float = 0.5,
    ):
        super().__init__()
        self.__diagonal = diagonal
        self.__p = p
        self.__portion_interval = portion_interval
        pass

    def _augment(self, img: Tensor, fill: ColorType) -> Tensor:
        if random.random() > self.__p:
            return img
        portion_size_percentage = (
            self.__portion_interval[0]
            + (self.__portion_interval[1] - self.__portion_interval[0])
            * random.random()
        )
        portion_size = int(img.shape[-1] * portion_size_percentage)
        is_side = random.random() > 0.5
        if not is_side:
            r = random.uniform(-90, 90)
            img = F.rotate(img, r, expand=True, fill=[fill[0], fill[1], fill[2]])
        which_side = random.random()
        if which_side < 0.25:  # TOP
            img = img[..., :portion_size, :]
        elif which_side < 0.50:  # BOTTOM
            img = img[..., img.shape[-1] - portion_size :, :]
        elif which_side < 0.75:  # LEFT
            img = img[..., :, :portion_size]
        else:  # RIGHT
            img = img[..., :, img.shape[-1] - portion_size :]
        return torch.nn.functional.interpolate(
            img.unsqueeze(0), (self.__diagonal, self.__diagonal), mode="bilinear"
        ).squeeze(0)


class RandomScalerDynamicFillerColor(DynamicColorAugmentators):
    def __init__(self, min_pad=0.0, max_pad=0.3, diagonal=400, p: float = 0.5):
        super().__init__()
        self.__min_pad = min_pad * diagonal
        self.__max_pad = max_pad * diagonal
        self.__diagonal = diagonal
        self.__p = p

    def _augment(self, img: Tensor, fill: ColorType):
        if random.random() > self.__p:
            return img
        r = random.random()
        pad = int((self.__max_pad - self.__min_pad) * r + self.__min_pad)
        padder = transforms.Pad(padding=pad, fill=fill, padding_mode="constant")
        img = padder(img)
        return torch.nn.functional.interpolate(
            img.unsqueeze(0), (self.__diagonal, self.__diagonal), mode="bilinear"
        ).squeeze(0)


class RandomFishEye:
    def __init__(self, p: float, height: int, width: int, magnitude: float = 0.25):
        self.__h = height
        self.__w = width
        self.__magnitude = magnitude
        self.__p = p

    def __call__(self, img: Tensor) -> Tensor:
        if random.random() > self.__p:
            return img
        choosen_center = torch.tensor(self.__compute_center())
        xx, yy = torch.linspace(-1, 1, self.__w), torch.linspace(-1, 1, self.__h)
        gridy, gridx = torch.meshgrid(yy, xx)  # create identity grid
        grid = torch.stack([gridx, gridy], dim=-1)
        d = choosen_center - grid  # calculate the distance(cx - x, cy - y)
        d_sum = torch.sqrt((d**2).sum(dim=-1))  # sqrt((cx-x)^2+(cy-y)^2)
        grid += d * d_sum.unsqueeze(-1) * self.__magnitude
        return torch.nn.functional.grid_sample(
            img.unsqueeze(0), grid.unsqueeze(0), align_corners=False
        ).squeeze(0)

    def __compute_center(self) -> Tuple[float, float]:
        computed_rand_center_x = random.random() * 0.2 - 0.1
        computed_rand_center_y = random.random() * 0.2 - 0.1
        return computed_rand_center_x, computed_rand_center_y


class RandomizeAugmentator:
    def __init__(self, augmentator: Callable, p: float) -> None:
        self.__aug = augmentator
        self.__p = p

    def __call__(self, img: Tensor) -> Tensor:
        return img if self.__p < random.random() else self.__aug(img)


class AugmentatorIf:
    def __init__(self, p: float, then: Callable, otherwise: Callable) -> None:
        self.__then = then
        self.__otherwise = otherwise
        self.__p = p

    def __call__(self, img: Tensor) -> Tensor:
        return self.__then(img) if random.random() > self.__p else self.__otherwise(img)


class RandomWaveDistortion:

    def __init__(self, height: int, width: int, curvature: float = 0.2, p=0.5):
        """
        Apply a single large bending curve effect, as if the image were attached to a large pipe.

        Args:
            p (float): Probability of applying the transformation.
            height (int): Image height.
            width (int): Image width.
            curvature (float): Controls the intensity of the bending effect.
        """
        self.__h = height
        self.__w = width
        self.__curvature = curvature  # Strength of the bend
        self.__p = p

    def __call__(self, img: Tensor) -> Tensor:
        """
        Apply the pipe bending effect to the image with probability `p`.

        Args:
            img (Tensor): Image tensor of shape [C, H, W].

        Returns:
            Tensor: Distorted image tensor.
        """
        if random.random() > self.__p:
            return img

        # Create a normalized coordinate grid [-1, 1]
        yy, xx = torch.linspace(-1, 1, self.__h), torch.linspace(-1, 1, self.__w)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # Shape: [H, W, 2]

        # Apply a uniform pipe bending effect (constant displacement across the entire height)
        fun_to_apply = torch.cos
        value_to_add = torch.pi / random.random()
        if random.random() < 0.5:
            fun_to_apply = torch.sin
        grid[..., 0] += self.__curvature * fun_to_apply(
            torch.pi * grid[..., 1] / 2 + value_to_add
        )  # Sinusoidal displacement

        # Apply grid_sample to warp the image
        return torch.nn.functional.grid_sample(
            img.unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
