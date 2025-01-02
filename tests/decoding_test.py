from shurikode import enc, dec

import pytest
import shurikode
import shurikode.shurikode_decoder
import shurikode.shurikode_encoder


indices = list(range(256))


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
    decoded_value = decoder(number_img)
    bit_list = [0] * 8
    idx = -1
    while value:
        bit_list[idx] = 1 & value
        value = value >> 1
        idx -= 1
    for p, gt in zip(decoded_value, bit_list):
        assert p == gt
