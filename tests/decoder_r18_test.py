from shurikode import Enc, Dec
import torchvision.transforms.v2 as transforms
import torch

import pytest
import random

IMAGES_DIAGONAL = 400

indices = list(range(256))

image_tensorizer = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(IMAGES_DIAGONAL),
    ]
)

augmentators_1 = transforms.Pad(
    padding=8,
    fill=[
        random.random(),
        random.random(),
        random.random(),
    ],
    padding_mode="constant",
)
augmentators_2 = transforms.GaussianBlur(25, 10)


@pytest.fixture(scope="module")
def encoder():
    return Enc(10)


@pytest.fixture(scope="module")
def decoder_r18():
    return Dec("S")


@pytest.mark.parametrize("value", indices)
def test_decoding_r18(
    value: int,
    encoder: Enc,
    decoder_r18: Dec,
):
    number_img = encoder.encode(value).get_PIL_image().convert("RGB")
    decoded_value_0 = decoder_r18(number_img)
    tensor_img: torch.Tensor = image_tensorizer(number_img).unsqueeze(0)
    tensor_img = torch.nn.functional.interpolate(
        tensor_img, (400, 400), mode="bilinear"
    )
    tensor_img = augmentators_1(tensor_img)
    tensor_img = torch.nn.functional.interpolate(
        tensor_img, (400, 400), mode="bilinear"
    )
    decoded_value_1 = decoder_r18(tensor_img)
    tensor_img = augmentators_2(tensor_img)
    decoded_value_2 = decoder_r18(tensor_img)
    assert (
        value == decoded_value_2
        and value == decoded_value_1
        and value == decoded_value_0
    )
