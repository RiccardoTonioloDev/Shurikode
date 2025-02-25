from torchvision import models
from torch import nn
from shurikode.utils import ModelType, DeviceType


import torch
from importlib.resources import files

model_selector = {
    "r18": ("ResNet18", models.resnet18),
    "r34": ("ResNet34", models.resnet34),
    "r50": ("ResNet50", models.resnet50),
}


def Create_ResNet_Shurikode(
    model_type: ModelType = "r50",
    out_features: int = 256,
    device: DeviceType = "cpu",
):
    """
    Given the model type, the output_features and a device on which load the model, returns the ResNet model configured
    in the desired way.

    :param model_type: The type of ResNet. Available choices are 'r18', 'r34' and 'r50'.
    :param out_features: The number of output features. Corresponds to the possible number of labels.
    :param device: The type of device on which to load the model. Available choices are 'cpu', 'cuda' and 'mps'.
    """
    str_model, func_model = model_selector[model_type]
    model = func_model()
    model.fc = nn.Linear(model.fc.in_features, out_features)
    checkpoint_path = files("shurikode.ml_model").joinpath(f"{str_model}.pth.tar")
    state_dict = torch.load(
        str(checkpoint_path), map_location=torch.device(device), weights_only=True
    )
    model.load_state_dict(state_dict)
    return model.to(device)
