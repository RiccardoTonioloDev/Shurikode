import torchvision.transforms.v2 as t
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple
from torch import Tensor
import pandas as pd
import torch
import os


class real_dataset(Dataset):
    tensorizer = t.Compose(
        [t.ToImage(), t.ToDtype(torch.float32, scale=True), t.Resize((400, 400))]
    )

    def __init__(self, images_folder: str) -> None:
        super().__init__()
        assert os.path.isdir(images_folder), "The images folder provided doesn't exist."
        self.images_folder = images_folder
        csv_path = os.path.join(images_folder, "SHURIKODE.csv")
        assert os.path.exists(csv_path), "The csv labeling file doesn't exist.."
        self.data_df = pd.read_csv(csv_path)
        self.data_df["Class"] = self.data_df["Class"].astype(int)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_file_name, img_class = self.data_df.iloc[index][["Img_Name", "Class"]]
        img_path = os.path.join(self.images_folder, img_file_name)
        assert os.path.exists(img_path), f"The image '{img_file_name}' doesn't exist."
        try:
            with Image.open(img_path) as img:
                image_tensor = self.tensorizer(img)
        except Exception as e:
            raise RuntimeError(
                f"Error: {e}\n\nCan't open image with the following path: {img_path}."
            )
        return image_tensor, img_class
