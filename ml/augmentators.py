from __future__ import annotations
import time
from torchvision.transforms.v2 import functional as F, InterpolationMode
from typing import Callable, List, Tuple
from abc import ABC, abstractmethod
from random import random
from torch import Tensor

import torch
import random


class ColorGenerator:
    """
    Generates colors so that each of its subscribers receives the same one, but it changes them every new iteration.
    """

    def __init__(self) -> None:
        self.__i = 0
        self.__fill = (random.random(), random.random(), random.random())
        self.__finished_subscribing = False
        self.__subscribers: List[object] = []

    def subscribe(self, subscriber: DynamicColorAugmentators) -> None:
        """
        Subscribes the subscriber to the color generator, in order for it to provide colors when asked.

        :param subscriber: The augmentator that will ask for colors using the `.get_colors()` method.
        """
        assert (
            not self.__finished_subscribing
        ), "After you called get_color for the first time, you can no longer \
            subscribe with other augmentators."
        self.__subscribers.append(subscriber)

    def get_color(
        self, color_asker: DynamicColorAugmentators
    ) -> Tuple[float, float, float]:
        """
        Returns the color of the current iteration.

        :param color_asker: The augmentator that is asking for the color of the current iteration.
        """
        assert (
            color_asker in self.__subscribers
        ), "The color asker must have previously subscribed to the color generator \
        in order to receive a color."
        self.__finished_subscribing = True
        sub_idx = self.__subscribers.index(color_asker)
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
    """
    An abstract class that will be used to create augmentators who will share a color generator.
    """

    def __init__(self) -> None:
        self.__color_gen = color_gen
        self.__color_gen.subscribe(self)

    def __call__(self, img: Tensor) -> Tensor:
        """
        Applies the augmentation to the given image.

        :param img: The image to augment.
        """
        fill = self.__color_gen.get_color(self)
        return self._augment(img, fill)

    @abstractmethod
    def _augment(self, img: Tensor, fill: ColorType) -> Tensor:
        """
        Applies the augmentation to the image, using the given filler color.

        :param img: The image to augment.
        :param fill: The tuple representing the color.
        """
        pass


class RandomRotationDynamicFillerColor(DynamicColorAugmentators):
    """
    Applies a random rotation (given an interval of degrees), to the given image.
    """

    def __init__(
        self, degrees: Tuple[int, int], diagonal: int = 400, expand=True, p: float = 0.5
    ):
        """
        :param degrees: The interval of degrees to be used for randomly choosing the rotation degree.
        :param diagonal: The number of pixels in the diagonal.
        :param expand: Wether to expand the augmented image in order to not loose information of the original one.
        :param p: The proability of the augmentation to take place.
        """
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
    """
    Applies a random change in perspective, to the given image.
    """

    def __init__(
        self,
        distortion_scale=0.5,
        interpolation=F.InterpolationMode.NEAREST,
        p: float = 0.5,
    ):
        """
        :param distortion_scale: The amount of distortion to be applied
        :param interpolation: The interpolation mode to be used.
        :param p: The proability of the augmentation to take place.
        """
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
    """
    Cuts a random portion of the image.
    """

    def __init__(
        self,
        diagonal=400,
        portion_interval: Tuple[float, float] = (0.5, 1.0),
        p: float = 0.5,
    ):
        """
        :param diagonal: The number of pixels in the diagonal.
        :param portion_interval: The interval of percentages of cut.
        :param p: The proability of the augmentation to take place.
        """
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
    """
    Adds a random padding to the given image.
    """

    def __init__(self, min_pad=0.0, max_pad=0.3, diagonal=400, p: float = 0.5):
        """
        :param min_pad: The minimum padding that will be able to perform.
        :param max_pad: The maximum padding that will be able to perform.
        :param diagonal: The number of pixels in the diagonal.
        :param p: The proability of the augmentation to take place.
        """
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
        img = F.pad(img, [pad, pad, pad, pad], [fill[0], fill[1], fill[2]], "constant")
        return torch.nn.functional.interpolate(
            img.unsqueeze(0), (self.__diagonal, self.__diagonal), mode="bilinear"
        ).squeeze(0)


class RandomizeAugmentator:
    """
    Gives the ability to randomize a non-random augmentator.
    """

    def __init__(self, augmentator: Callable, p: float) -> None:
        """
        :param augmentator: The augmentator to randomize.
        :param p: The proability of the augmentation to take place.
        """
        self.__aug = augmentator
        self.__p = p

    def __call__(self, img: Tensor) -> Tensor:
        """
        Returns the transformed image, if the augmentator was used, otherwise it maintains the image intact.
        """
        return img if self.__p < random.random() else self.__aug(img)


class AugmentatorIf:
    """
    Choose between two different augmentators, with the first having a p probability of being executed, and the second
    a 1-p probability.
    """

    def __init__(self, p: float, then: Callable, otherwise: Callable) -> None:
        """
        :param p: The proability of the augmentation to take place.
        :param then: The first augmentator.
        :param otherwise: The second augmentator.
        """
        self.__then = then
        self.__otherwise = otherwise
        self.__p = p

    def __call__(self, img: Tensor) -> Tensor:
        """
        Returns the transformed image, with the randomly chosen augmentation.
        """
        return self.__then(img) if random.random() > self.__p else self.__otherwise(img)


class RandomWaveDistortion:
    """
    Applies a single large bending curve effect, as if the image were attached to a large pipe.
    """

    def __init__(self, height: int, width: int, curvature: float = 0.2, p=0.5):
        """
        :param p: The proability of the augmentation to take place.
        :param height: Image height.
        :param width): Image width.
        :param curvature: Controls the intensity of the bending effect.
        """
        self.__h = height
        self.__w = width
        self.__curvature = curvature  # Strength of the bend
        self.__p = p

    def __call__(self, img: Tensor) -> Tensor:
        """
        Applies the pipe bending effect to the image with probability p.
        """
        if random.random() > self.__p:
            return img

        # Create a normalized coordinate grid [-1, 1]
        yy, xx = torch.linspace(-1, 1, self.__h), torch.linspace(-1, 1, self.__w)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # Shape: [H, W, 2]

        # Apply a uniform pipe bending effect (constant displacement across the entire height)
        fun_to_apply = torch.cos
        value_to_scale = 3 + random.random()
        if random.random() < 0.5:
            fun_to_apply = torch.sin
        grid[..., 0] += self.__curvature * fun_to_apply(
            torch.pi * grid[..., 1] / (value_to_scale)
        )  # Sinusoidal displacement

        # Apply grid_sample to warp the image
        return torch.nn.functional.grid_sample(
            img.unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)


class RandomBrightnessAdjust:
    """
    Randomly modifies the brightness, given an interval of brightness.
    """

    def __init__(self, brightness_interval=(0.8, 1.2), p=0.5):
        """
        :param brightness_interval: The interval of brightness that will be used.
        :param p: The proability of the augmentation to take place.
        """
        self.__bri_interval = brightness_interval
        self.__p = p

    def __call__(self, img: Tensor) -> Tensor:
        """
        Applies the brightness augmentation with probability p.
        """
        if random.random() > self.__p:
            return img

        brightness_factor = self.__bri_interval[0] + random.random() * (
            self.__bri_interval[1] - self.__bri_interval[0]
        )

        img = F.adjust_brightness_image(img, brightness_factor)
        return img
