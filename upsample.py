from torch import Tensor

import torch.nn as nn


class Upsample(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            use_conv: bool,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        return

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)

        if self.use_conv:
            x = self.conv(x)

        return x
