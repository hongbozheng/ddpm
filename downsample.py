from torch import Tensor

import torch.nn as nn


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )

        return

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)

        return x
