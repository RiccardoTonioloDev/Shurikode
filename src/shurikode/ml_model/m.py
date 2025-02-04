from torchvision import models
from torch import nn
from typing import Literal

import torch
from importlib.resources import files


def Create_ResNet50(device: Literal["cuda", "mps", "cpu"]):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 256)
    checkpoint_path = files("shurikode.ml_model").joinpath("ResNet50.pth.tar")
    state_dict = torch.load(
        str(checkpoint_path), map_location=torch.device(device), weights_only=True
    )
    model.load_state_dict(state_dict)
    return model


def Create_ResNet18(device: Literal["cuda", "mps", "cpu"]):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 256)
    checkpoint_path = files("shurikode.ml_model").joinpath("ResNet18.pth.tar")
    state_dict = torch.load(
        str(checkpoint_path), map_location=torch.device(device), weights_only=True
    )
    model.load_state_dict(state_dict)
    return model
