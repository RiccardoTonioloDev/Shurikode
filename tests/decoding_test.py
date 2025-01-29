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
# augmentators_2 = transforms.GaussianBlur(25, 10)


@pytest.fixture(scope="module")
def encoder():
    return enc(10)


@pytest.fixture(scope="module")
def decoder():
    return dec()


@pytest.mark.parametrize("value", indices)
def test_decoding_of(
    value: int,
    encoder: shurikode.shurikode_encoder.shurikode_encoder,
    decoder: shurikode.shurikode_decoder.shurikode_decoder,
):
    number_img = encoder.encode(value).get_PIL_image()
    tensor_img: torch.Tensor = image_tensorizer(number_img).unsqueeze(0)
    tensor_img = torch.nn.functional.interpolate(
        tensor_img, (400, 400), mode="bilinear"
    )
    tensor_img = augmentators_1(tensor_img)
    # tensor_img = torch.nn.functional.interpolate(
    #     tensor_img, (400, 400), mode="bilinear"
    # )
    # tensor_img = augmentators_2(tensor_img)
    decoded_value = decoder(number_img)
    assert value == decoded_value
