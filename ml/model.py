from torch import nn
from torch import Tensor
from attention_modules import CBAM
from typing import List, Tuple

import torch


def xavier_init(m: torch.nn.Module):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.trunc_normal_(m.bias)


class BinNet_alexnet(nn.Module):
    def __init__(self):
        super(BinNet_alexnet, self).__init__()

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

        # with torch.no_grad():
        #     CodeExtractor.__binary_transformer = torch.zeros(
        #         (8, 256), dtype=torch.float32
        #     )
        #     for i in range(256):
        #         binary_code = [int(bit) for bit in format(i, "08b")]
        #         CodeExtractor.__binary_transformer[:, i] = torch.tensor(binary_code)

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
        # return torch.matmul(self.__layers(x), self.__binary_transformer.detach())
        return self.__layers(x)


class BinNet_stair(nn.Module):
    def __init__(self):
        super(BinNet_stair, self).__init__()

        self.__level_1_starter, nf_l1s = BinNet_stair.__start_level_downscaler(
            1
        )  # output: [16, 400, 400]
        self.__level_2_starter, nf_l2s = BinNet_stair.__start_level_downscaler(
            2
        )  # output: [32, 200, 200]
        self.__level_3_starter, nf_l3s = BinNet_stair.__start_level_downscaler(
            3
        )  # output: [64, 100, 100]
        self.__level_4_starter, nf_l4s = BinNet_stair.__start_level_downscaler(
            4
        )  # output: [128, 50, 50]

        self.__level_1_enhancer, nf_l1e = BinNet_stair.__block_level_enhancer(
            nf_l1s
        )  # output: [256, 400, 400]
        self.__level_2_enhancer, nf_l2e = BinNet_stair.__block_level_enhancer(
            nf_l2s
        )  # output: [256, 200, 200]
        self.__level_3_enhancer, nf_l3e = BinNet_stair.__block_level_enhancer(
            nf_l3s
        )  # output: [256, 100, 100]
        self.__level_4_enhancer, nf_l4e = BinNet_stair.__block_level_enhancer(
            nf_l4s
        )  # output: [256, 50, 50]

        self.__level_1_downscaler, nf_l1d = BinNet_stair.__end_level_downscaler(
            nf_l1e
        )  # output: [256, 200, 200]
        self.__level_2_downscaler, nf_l2d = BinNet_stair.__end_level_downscaler(
            nf_l2e
        )  # output: [256, 100, 100]
        self.__level_3_downscaler, nf_l3d = BinNet_stair.__end_level_downscaler(
            nf_l3e
        )  # output: [256, 50, 50]

        self.__level_2_compressor, nf_l2c = BinNet_stair.__end_level_compressor(
            nf_l1d + nf_l2e, int((nf_l1d + nf_l2e) / 2)
        )  # output: [256, 200, 200]
        self.__level_3_compressor, nf_l3c = BinNet_stair.__end_level_compressor(
            nf_l2c + nf_l3e, int((nf_l2c + nf_l3e) / 2)
        )  # output: [256, 100, 100]
        self.__level_4_compressor, nf_l4c = BinNet_stair.__end_level_compressor(
            nf_l3c + nf_l4e, int((nf_l3c + nf_l4e) / 2)
        )  # output: [256, 50, 50]

        self.__final_prediction_head = BinNet_stair.__prediction_head(256 * 50 * 50, 8)

        self.apply(xavier_init)

    def training_forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Downscaling
        x1 = self.__level_1_starter(x)
        x2 = self.__level_2_starter(x1)
        x3 = self.__level_3_starter(x2)
        x4 = self.__level_4_starter(x3)

        # Processing
        x1 = self.__level_1_enhancer(x1)
        x2 = self.__level_2_enhancer(x2)
        x3 = self.__level_3_enhancer(x3)
        x4 = self.__level_4_enhancer(x4)

        # Downscaling
        x1_downscale_l2 = self.__level_1_downscaler(x1)
        x2x1_to_compress = torch.cat([x1_downscale_l2, x2], 1)
        x2x1_compressed = self.__level_2_compressor(x2x1_to_compress)
        x1_downscale_l4 = self.__level_3_downscaler(
            self.__level_2_downscaler(x1_downscale_l2)
        )

        x2x1_downscale_l3 = self.__level_2_downscaler(x2x1_compressed)
        x3x2x1_to_compress = torch.cat([x2x1_downscale_l3, x3], 1)
        x3x2x1_compressed = self.__level_3_compressor(x3x2x1_to_compress)
        x2_downscale_l4 = self.__level_3_downscaler(self.__level_2_downscaler(x2))

        x3x2x1_downscale_l4 = self.__level_3_downscaler(x3x2x1_compressed)
        x4x3x2x1_to_compress = torch.cat([x3x2x1_downscale_l4, x4], 1)
        x4x3x2x1_compressed = self.__level_4_compressor(x4x3x2x1_to_compress)
        x3_downscale_l4 = self.__level_3_downscaler(x3)

        return (
            self.__final_prediction_head(x4x3x2x1_compressed),
            self.__final_prediction_head(x4),
            self.__final_prediction_head(x3_downscale_l4),
            self.__final_prediction_head(x2_downscale_l4),
            self.__final_prediction_head(x1_downscale_l4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downscaling
        print("downscaler block:")
        x1 = self.__level_1_starter(x)
        print(x1.shape)
        x2 = self.__level_2_starter(x1)
        print(x2.shape)
        x3 = self.__level_3_starter(x2)
        print(x3.shape)
        x4 = self.__level_4_starter(x3)
        print(x4.shape)

        # Processing
        print("enhancer block:")
        x1 = self.__level_1_enhancer(x1)
        print(x1.shape)
        x2 = self.__level_2_enhancer(x2)
        print(x2.shape)
        x3 = self.__level_3_enhancer(x3)
        print(x3.shape)
        x4 = self.__level_4_enhancer(x4)
        print(x4.shape)

        # Downscaling
        x1_downscale_l2 = self.__level_1_downscaler(x1)
        x2x1_to_compress = torch.cat([x1_downscale_l2, x2], 1)
        x2x1_compressed = self.__level_2_compressor(x2x1_to_compress)

        x2x1_downscale_l3 = self.__level_2_downscaler(x2x1_compressed)
        x3x2x1_to_compress = torch.cat([x2x1_downscale_l3, x3], 1)
        x3x2x1_compressed = self.__level_3_compressor(x3x2x1_to_compress)

        x3x2x1_downscale_l4 = self.__level_3_downscaler(x3x2x1_compressed)
        x4x3x2x1_to_compress = torch.cat([x3x2x1_downscale_l4, x4], 1)
        x4x3x2x1_compressed = self.__level_4_compressor(x4x3x2x1_to_compress)

        return self.__final_prediction_head(x4x3x2x1_compressed)

    @staticmethod
    def __start_level_downscaler(level: int) -> Tuple[nn.Module, int]:
        assert level >= 1, "The level parameter has to be greater than 1."
        layers: List[nn.Module] = [
            nn.Conv2d(
                3 if level == 1 else 2 ** (3 + level - 1),
                2 ** (3 + level),
                3,
                padding=1,
                stride=1 if level == 1 else 2,
            ),
            nn.BatchNorm2d(2 ** (3 + level)),
            nn.LeakyReLU(0.1),
            CBAM(2 ** (3 + level)),
        ]

        return nn.Sequential(*layers), 2 ** (3 + level)

    @staticmethod
    def __block_level_enhancer(input_features: int) -> Tuple[nn.Module, int]:
        layers: List[nn.Module] = []
        cycles = 2
        for i in range(cycles):
            layers.append(
                nn.Conv2d(
                    input_features if i == 0 else 2 ** (8 + i - 2),
                    2 ** (8 + i - 1),
                    3,
                    padding=1,
                ),
            )
            layers.append(nn.BatchNorm2d(2 ** (8 + i - 1)))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(CBAM(2 ** (8 + i - 1)))
        return nn.Sequential(*layers), 2 ** (8 + cycles - 2)

    @staticmethod
    def __end_level_downscaler(input_features: int) -> Tuple[nn.Module, int]:
        return nn.Conv2d(input_features, input_features, 3, 2, 1), input_features

    @staticmethod
    def __end_level_compressor(
        input_features: int, output_features: int
    ) -> Tuple[nn.Module, int]:
        return nn.Conv2d(input_features, output_features, 1), output_features

    @staticmethod
    def __prediction_head(input_features: int, output_features: int):
        return nn.Sequential(
            nn.Flatten(), nn.Linear(input_features, output_features), nn.Sigmoid()
        )


if __name__ == "__main__":
    m = BinNet_alexnet()
    x = torch.ones([5, 3, 400, 400])
    print(m(x).shape)
    print(
        f"Number of parameters (alexnet): {sum(p.numel() for p in m.parameters() if p.requires_grad)}"
    )

    m = BinNet_stair()
    x = torch.rand([1, 3, 400, 400])
    m(x)
    m.training_forward(x)
    print(
        f"Number of parameters (stairs): {sum(p.numel() for p in m.parameters() if p.requires_grad)}"
    )
