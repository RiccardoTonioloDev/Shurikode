from shurikode import enc, dec
import torchvision.transforms.v2 as transforms
import torch

import pytest
import shurikode
import shurikode.shurikode_decoder
import shurikode.shurikode_encoder


indices = list(range(256))

image_tensorizer = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=False),
    ]
)


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
    tensor_img = torch.nn.functional.pad(tensor_img, (60, 60, 60, 60), "constant", 0)
    decoded_value = decoder(number_img)
    assert value == decoded_value
