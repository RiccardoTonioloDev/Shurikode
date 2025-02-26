from shurikode import enc, dec
import torchvision.transforms.v2 as transforms
import torch

import pytest
import shurikode
import random
import shurikode.shurikode_decoder
import shurikode.shurikode_encoder


indices = list(range(256))

image_tensorizer = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
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
    return enc(10)


@pytest.fixture(scope="module")
def decoder_r50():
    return dec("r50")


@pytest.mark.parametrize("value", indices)
def test_decoding_r50(
    value: int,
    encoder: shurikode.shurikode_encoder.shurikode_encoder,
    decoder_r50: shurikode.shurikode_decoder.shurikode_decoder,
):
    number_img = encoder.encode(value).get_PIL_image()
    decoded_value_0 = decoder_r50(number_img)
    tensor_img: torch.Tensor = image_tensorizer(number_img).unsqueeze(0)
    tensor_img = torch.nn.functional.interpolate(
        tensor_img, (400, 400), mode="bilinear"
    )
    tensor_img = augmentators_1(tensor_img)
    tensor_img = torch.nn.functional.interpolate(
        tensor_img, (400, 400), mode="bilinear"
    )
    decoded_value_1 = decoder_r50(tensor_img)
    tensor_img = augmentators_2(tensor_img)
    decoded_value_2 = decoder_r50(tensor_img)
    assert (
        value == decoded_value_2
        and value == decoded_value_1
        and value == decoded_value_0
    )
