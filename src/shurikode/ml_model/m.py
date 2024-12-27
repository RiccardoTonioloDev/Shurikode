from torchvision import models
from torch import nn
from typing import Literal

import torch
import os


def Create_ResNet50(device: Literal["cuda", "mps", "cpu"]):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 8)
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    checkpoint_path = os.path.join(current_dir, "ResNet50.pth.tar")
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    return model
