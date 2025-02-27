import math
from typing import Literal
from torch import nn
from torchvision import models


ResNetType = Literal["18", "34", "50"]

model_selector = {
    "r18": (models.ResNet18_Weights.IMAGENET1K_V1, models.resnet18),
    "r34": (models.ResNet34_Weights.IMAGENET1K_V1, models.resnet34),
    "r50": (models.ResNet50_Weights.IMAGENET1K_V2, models.resnet50),
}


def Create_ResNet(
    resnet_type: ResNetType,
    weights=None,
    n_classes=256,
    binary_output=False,
    hamming_correction_bits=0,
) -> nn.Module:
    assert (
        not binary_output and hamming_correction_bits != 0
    ), "You can't apply hamming correction on a model with a non-binary output."
    if binary_output:
        n_classes = math.ceil(math.log(n_classes, 2)) + hamming_correction_bits
    assert (
        2**hamming_correction_bits
    ) >= n_classes + hamming_correction_bits + 1, "The hamming correction bits should satisfy this formula: \
        2^#hamm_bits >= #data_bits + #hamm_bits + 1. This formula is currenly not satisfied"
    w = weights
    if not w:
        w = model_selector[resnet_type][0]

    model: nn.Module = model_selector[resnet_type][1](weights=w)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    if weights:
        model.load_state_dict(weights)

    return model
