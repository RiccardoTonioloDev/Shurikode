from torch import nn
from torch import Tensor
from ml.attention_modules import CBAM

import torch


class CodeExtractor(nn.Module):
    def __init__(self):
        super(CodeExtractor, self).__init__()

        self.__layers = nn.Sequential(
            self.conv_block(3, 16, 7),
            CBAM(16),
            self.downsample_block(16, 32, 3),
            self.conv_block(32, 64, 3),
            CBAM(64),
            self.downsample_block(64, 128, 3),
            CBAM(128),
            self.downsample_block(128, 256, 3),
            CBAM(256),
            self.downsample_block(256, 512, 3),
            CBAM(512),
            self.downsample_block(512, 128, 3),
            CBAM(128),
            nn.Flatten(),
            nn.Linear(21632, 256),
        )

    @staticmethod
    def conv_block(in_chann: int, out_chann: int, kernel: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_chann, out_chann, kernel, padding=kernel // 2),
            nn.BatchNorm2d(out_chann),
            nn.LeakyReLU(0.1),
        )

    @staticmethod
    def downsample_block(in_chann: int, out_chann: int, kernel: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_chann, out_chann, kernel, stride=2, padding=kernel // 2),
            nn.BatchNorm2d(out_chann),
            nn.LeakyReLU(0.1),
        )

    def forward_training(self, x: Tensor) -> Tensor:
        return self.__layers(x)

    def forward(self, x: Tensor) -> Tensor:
        return torch.softmax(self.__layers(x), 1)


m = CodeExtractor()
x = torch.ones([1, 3, 400, 400])
print(m(x).shape)
print(
    f"Number of parameters: {sum(p.numel() for p in m.parameters() if p.requires_grad)}"
)
