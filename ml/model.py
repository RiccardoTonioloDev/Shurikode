import math
from typing import Literal
from torch import nn
from torchvision import models


ResNetType = Literal["r18", "r34", "r50"]

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
    hamming_output=False,
) -> nn.Module:
    """
    Creates a ResNet model of the selected type and configuration.

    :param resnet_type: The type of resnet model chosen.
    :param weights: The weights, previously learned, to be used in the created model.
    :param n_classes: The number of classes supported by the model.
    :param resnet_type: Wether to use a binary encoding or a sparse vector encoding, for the model output.
    :param hamming_output: Wether the model considers also hamming correction bits, in the model output.
    """
    assert (
        binary_output or not hamming_output
    ), "You can't apply hamming correction on a model with a non-binary output."

    output_features = n_classes
    if binary_output:
        output_features = math.ceil(math.log(n_classes, 2))
        if hamming_output:
            correction_bits = 0
            # Determine the number of parity bits needed
            while (2**correction_bits) < (n_classes + correction_bits + 1):
                correction_bits += 1
            output_features += correction_bits
    w = weights
    if not w:
        w = model_selector[resnet_type][0]

    model: nn.Module = model_selector[resnet_type][1](weights=w)
    model.fc = nn.Linear(model.fc.in_features, output_features)

    if weights:
        model.load_state_dict(weights)

    return model
