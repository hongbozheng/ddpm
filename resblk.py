from torch import Tensor

import torch.nn as nn


class ResBlk(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_groups: int,
            t_emb: int,
    ) -> None:
        super().__init__()
        self.blk_0 = nn.Sequential(
            nn.GroupNorm(num_groups=n_groups, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.t_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=t_emb, out_features=out_channels),
        )
        self.blk_1 = nn.Sequential(
            nn.GroupNorm(num_groups=n_groups, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.skip_conv = nn.Identity()

        return

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        _x = self.skip_conv(x)

        # [N, IC, H, W] -> [N, OC, H, W]
        x = self.blk_0(x)

        x += self.t_emb_proj(t)[:, :, None, None]

        # [N, OC, H, W] -> [N, OC, H, W]
        x = self.blk_1(x)

        x += _x

        return x
