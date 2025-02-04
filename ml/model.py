from torch import nn
from torch import Tensor
from attention_modules import CBAM
from typing import List, Tuple
from torchvision import models

import torch


def xavier_init(m: torch.nn.Module):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.trunc_normal_(m.bias)


class AlexNetAlike(nn.Module):
    def __init__(self):
        super(AlexNetAlike, self).__init__()

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
            nn.Linear(21632, 1000),
            nn.Dropout(),
            nn.Linear(1000, 8),
            nn.Sigmoid(),
        )

        self.apply(xavier_init)

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

    def forward(self, x: Tensor) -> Tensor:
        return self.__layers(x)


class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()

        self.__layers = nn.Sequential(
            self.conv_block(3, 16, 7),
            self.conv_block(16, 32, 3),
            self.conv_block(32, 64, 3),
            self.conv_block(64, 128, 3),
            self.conv_block(128, 256, 3),
            self.downsample_block(256, 512, 3),
            self.downsample_block(512, 512, 3),
            self.downsample_block(512, 256, 3),
            self.downsample_block(256, 128, 3),
            self.conv_block(128, 64, 3, cbam=True),
            self.conv_block(64, 32, 3, cbam=True),
            nn.Flatten(),
            nn.Linear(20000, 256),
            nn.Dropout(),
            nn.Linear(256, 8),
            nn.Sigmoid(),
        )

        self.apply(xavier_init)

    @staticmethod
    def conv_block(
        in_chann: int, out_chann: int, kernel: int, stride=1, cbam=False
    ) -> nn.Module:
        layers: List[nn.Module] = [
            nn.Conv2d(in_chann, out_chann, kernel, padding=kernel // 2, stride=stride),
            nn.BatchNorm2d(out_chann),
            nn.LeakyReLU(0.1),
        ]
        if cbam:
            layers.append(CBAM(out_chann))
        return nn.Sequential(*layers)

    @staticmethod
    def downsample_block(in_chann: int, out_chann: int, kernel: int) -> nn.Module:
        return DeepConvNet.conv_block(in_chann, out_chann, kernel, stride=2, cbam=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.__layers(x)
        return x


class DeepStair(nn.Module):
    def __init__(self):
        super(DeepStair, self).__init__()

        self.__l1_downscaler, nf_l1s = DeepStair.__downscaler(
            1
        )  # output: [16, 400, 400]
        self.__l2_downscaler, nf_l2s = DeepStair.__downscaler(
            2
        )  # output: [32, 200, 200]
        self.__l3_downscaler, nf_l3s = DeepStair.__downscaler(
            3
        )  # output: [64, 100, 100]
        self.__l4_downscaler, nf_l4s = DeepStair.__downscaler(
            4
        )  # output: [128, 50, 50]

        self.__l1_encoder, nf_l1e = DeepStair.__encoder(
            nf_l1s
        )  # output: [256, 400, 400]
        self.__l2_encoder, nf_l2e = DeepStair.__encoder(
            nf_l2s + nf_l1e
        )  # output: [256, 200, 200]
        self.__l3_encoder, nf_l3e = DeepStair.__encoder(
            nf_l3s + nf_l2e
        )  # output: [256, 100, 100]
        self.__l4_encoder, nf_l4e = DeepStair.__encoder(
            nf_l4s + nf_l3e
        )  # output: [256, 50, 50]

        self.__final_prediction_head = DeepStair.__prediction_head(16 * 25 * 25, 8)

        self.apply(xavier_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__l1_downscaler(x)
        x2 = self.__l2_downscaler(x)

        x = self.__l1_encoder(x)

        x3 = self.__l3_downscaler(x2)

        x = self.__l2_encoder(self.__concat(x2, x))
        del x2

        x4 = self.__l4_downscaler(x3)

        x = self.__l3_encoder(self.__concat(x3, x))
        del x3

        x = self.__l4_encoder(self.__concat(x4, x))
        del x4

        return self.__final_prediction_head(x)

    @staticmethod
    def __downscaler(level: int) -> Tuple[nn.Module, int]:
        assert level >= 1, "The level parameter has to be greater than 1."
        layers: List[nn.Module] = [
            DeepStair.__Conv2d_Block(
                3 if level == 1 else 2 ** (3 + level - 1),
                2 ** (3 + level),
                3,
                padding=1,
                stride=1 if level == 1 else 2,
            ),
        ]

        return nn.Sequential(*layers), 2 ** (3 + level)

    @staticmethod
    def __encoder(input_features: int) -> Tuple[nn.Module, int]:
        return (
            nn.Sequential(
                DeepStair.__Conv2d_Block(input_features, 512, 3, padding=1),
                DeepStair.__Conv2d_Block(512, 128, 3, padding=1),
                DeepStair.__Conv2d_Block(128, 32, 3, padding=1),
                DeepStair.__Conv2d_Block(32, 16, 3, padding=1),
                DeepStair.__Conv2d_Block(16, 16, 3, padding=1, stride=2),
            ),
            16,
        )

    @staticmethod
    def __concat(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return torch.cat([t1, t2], 1)

    @staticmethod
    def __prediction_head(input_features: int, output_features: int):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 1000),
            nn.Dropout(),
            nn.Linear(1000, output_features),
            nn.Sigmoid(),
        )

    @staticmethod
    def __Conv2d_Block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
    ) -> nn.Module:

        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            CBAM(out_channels),
        )


class ShortDeepStair(nn.Module):
    def __init__(self):
        super(ShortDeepStair, self).__init__()

        self.__l1_downscaler, nf_l1s = ShortDeepStair.__downscaler(
            1
        )  # output: [16, 400, 400]
        self.__l2_downscaler, nf_l2s = ShortDeepStair.__downscaler(
            2
        )  # output: [32, 200, 200]
        self.__l3_downscaler, nf_l3s = ShortDeepStair.__downscaler(
            3
        )  # output: [64, 100, 100]
        self.__l4_downscaler, nf_l4s = ShortDeepStair.__downscaler(
            4
        )  # output: [128, 50, 50]

        self.__l2_encoder, nf_l2e = ShortDeepStair.__encoder(
            nf_l2s
        )  # output: [256, 200, 200]
        self.__l3_encoder, nf_l3e = ShortDeepStair.__encoder(
            nf_l3s + nf_l2e
        )  # output: [256, 100, 100]
        self.__l4_encoder, nf_l4e = ShortDeepStair.__encoder(
            nf_l4s + nf_l3e
        )  # output: [256, 50, 50]

        self.__final_prediction_head = ShortDeepStair.__prediction_head(16 * 25 * 25, 8)

        self.apply(xavier_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__l2_downscaler(self.__l1_downscaler(x))

        x3 = self.__l3_downscaler(x)

        x = self.__l2_encoder(x)

        x4 = self.__l4_downscaler(x3)

        x = self.__l3_encoder(self.__concat(x3, x))
        del x3

        x = self.__l4_encoder(self.__concat(x4, x))
        del x4

        return self.__final_prediction_head(x)

    @staticmethod
    def __downscaler(level: int) -> Tuple[nn.Module, int]:
        assert level >= 1, "The level parameter has to be greater than 1."
        layers: List[nn.Module] = [
            ShortDeepStair.__Conv2d_Block(
                3 if level == 1 else 2 ** (3 + level - 1),
                2 ** (3 + level),
                3,
                padding=1,
            ),
            ShortDeepStair.__Conv2d_Block(
                2 ** (3 + level),
                2 ** (3 + level),
                3,
                padding=1,
                stride=1 if level == 1 else 2,
            ),
        ]

        return nn.Sequential(*layers), 2 ** (3 + level)

    @staticmethod
    def __encoder(input_features: int) -> Tuple[nn.Module, int]:
        return (
            nn.Sequential(
                ShortDeepStair.__Conv2d_Block(input_features, 256, 3, padding=1),
                ShortDeepStair.__Conv2d_Block(256, 512, 3, padding=1),
                ShortDeepStair.__Conv2d_Block(512, 128, 3, padding=1),
                ShortDeepStair.__Conv2d_Block(128, 32, 3, padding=1),
                ShortDeepStair.__Conv2d_Block(32, 16, 3, padding=1),
                ShortDeepStair.__Conv2d_Block(16, 16, 3, padding=1, stride=2),
            ),
            16,
        )

    @staticmethod
    def __concat(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return torch.cat([t1, t2], 1)

    @staticmethod
    def __prediction_head(input_features: int, output_features: int):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 1000),
            nn.Dropout(),
            nn.Linear(1000, output_features),
            nn.Sigmoid(),
        )

    @staticmethod
    def __Conv2d_Block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
    ) -> nn.Module:

        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            CBAM(out_channels),
        )


def Create_ResNet50_binary(weights=None, out_features=8):
    model = None
    if not weights:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, out_features)
    if weights:
        model.load_state_dict(weights)
    return model


def Create_ResNet50_prob_vec(weights=None, n_classes=256):
    model = None
    if not weights:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    if weights:
        model.load_state_dict(weights)
    return model


def Create_ResNet34_prob_vec(weights=None, n_classes=256):
    model = None
    if not weights:
        model = models.resnet34(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet34()
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    if weights:
        model.load_state_dict(weights)
    return model


def Create_ResNet18_prob_vec(weights=None, n_classes=256):
    model = None
    if not weights:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    if weights:
        model.load_state_dict(weights)
    return model


if __name__ == "__main__":
    m = AlexNetAlike()
    x = torch.ones([5, 3, 400, 400])
    print(
        f"Number of parameters (alexnet): {sum(p.numel() for p in m.parameters() if p.requires_grad)}"
    )

    m = DeepConvNet()
    x = torch.rand([1, 3, 400, 400])
    m(x)
    print(
        f"Number of parameters (deepconvnet): {sum(p.numel() for p in m.parameters() if p.requires_grad)}"
    )

    m = DeepStair()
    x = torch.rand([1, 3, 400, 400])
    m(x)
    print(
        f"Number of parameters (deepstairs): {sum(p.numel() for p in m.parameters() if p.requires_grad)}"
    )

    m = ShortDeepStair()
    x = torch.rand([1, 3, 400, 400])
    m(x)
    print(
        f"Number of parameters (shortdeepstairs): {sum(p.numel() for p in m.parameters() if p.requires_grad)}"
    )

    m = Create_ResNet50_binary()
    x = torch.rand([1, 3, 400, 400])
    print(m(x))
    print(
        f"Number of parameters (shortdeepstairs): {sum(p.numel() for p in m.parameters() if p.requires_grad)}"
    )
