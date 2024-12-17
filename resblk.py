from torch import Tensor

import torch.nn as nn
from downsample import Downsample
from upsample import Upsample


class ResBlk(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            t_emb_dim: int,
            down: bool,
            up: bool,
            n_groups: int,
            eps: float,
    ) -> None:
        super().__init__()
        self.norm_0 = nn.GroupNorm(
            num_groups=n_groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
        )
        self.silu = nn.SiLU()
        self.conv_0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.upsample = self.downsample = None

        if down:
            self.downsample = Downsample(
                in_channels=in_channels,
                out_channels=in_channels,
                use_conv=False,
            )
        elif up:
            self.upsample = Upsample(
                in_channels=in_channels,
                out_channels=in_channels,
                use_conv=False,
            )

        self.t_emb_proj = nn.Linear(
            in_features=t_emb_dim,
            out_features=out_channels,
        )

        self.norm_1 = nn.GroupNorm(
            num_groups=n_groups,
            num_channels=out_channels,
            eps=eps,
            affine=True,
        )

        self.conv_1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if in_channels != out_channels:
            self.skip_connect = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.skip_connect = nn.Identity()

        return

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        _x = self.skip_connect(x)

        # [N, IC, H, W] -> [N, OC, H, W]
        x = self.norm_0(x)
        x = self.silu(x)

        if self.downsample is not None:
            _x = self.downsample(_x)
            x = self.downsample(x)
        elif self.upsample is not None:
            _x = self.upsample(_x)
            x = self.upsample(x)

        x = self.conv_0(x)

        t = self.silu(t)
        # [N, T] -> [N, T, 1, 1]
        t_emb = self.t_emb_proj(t)[:, :, None, None]

        x += t_emb

        x = self.norm_1(x)
        x = self.silu(x)
        x = self.conv_1(x)

        x += _x

        return x
