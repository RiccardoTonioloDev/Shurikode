from typing import Dict, Any

import torch
import os


def save_model(
    checkpoints_path: str, name: str, model_state_dict: Dict[str, Any]
) -> str:
    checkpoint_filename = os.path.join(checkpoints_path, f"checkpoint_{name}.pth.tar")
    torch.save(model_state_dict, checkpoint_filename)
    return checkpoint_filename
