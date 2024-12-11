from torch import nn

import torch


class CBAM(nn.Module):
    """
    References:
        - Spatial attention: https://paperswithcode.com/method/spatial-attention-module
        - Channel attention: https://paperswithcode.com/method/channel-attention-module
        - 3D attention: https://joonyoung-cv.github.io/assets/paper/20_ijcv_a_simple.pdf
    """

    def __init__(self, in_channels: int, reduction_ratio=8):
        assert (
            in_channels >= 16
        ), "Input channels have to be greater than 16 for the attention module to work correctly"
        super(CBAM, self).__init__()
        #################################
        #             CHANNEL           #
        #################################
        self.channels_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
        )
        #################################
        #             SPATIAL           #
        #################################
        self.spacial_net = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
        )

    def forward(self, x: torch.Tensor):
        #################################
        #             CHANNEL           #
        #################################
        avg_pool = nn.functional.avg_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )
        max_pool = nn.functional.max_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )
        channel_att: torch.Tensor = self.channels_mlp(avg_pool) + self.channels_mlp(
            max_pool
        )
        channel_att = channel_att.sigmoid().unsqueeze(2).unsqueeze(3)
        residual_channel_att = x * channel_att
        #################################
        #             SPATIAL           #
        #################################
        compressed_x = torch.cat(
            (
                torch.max(residual_channel_att, 1)[0].unsqueeze(1),
                torch.mean(residual_channel_att, 1).unsqueeze(1),
            ),
            dim=1,
        )
        spacial_att: torch.Tensor = self.spacial_net(compressed_x).sigmoid()
        #################################
        #            COMBINED           #
        #################################
        residual_spacial_attention = x * spacial_att
        return residual_spacial_attention + x
