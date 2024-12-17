from torch import Tensor
from typing import List

import torch
import torch.nn as nn
from attention import MultiHeadAttention
from resblk import ResBlk
from upsample import Upsample


class UpBlk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channel: int,
        out_channels: int,
        t_emb_dim: int,
        n_blks: int,
        n_groups: int,
        eps: float,
        add_us: bool,
    ) -> None:
        super().__init__()

        res_blks = []

        for i in range(n_blks):
            res_skip_channels = in_channels if (i == n_blks - 1) else out_channels
            res_in_channels = prev_out_channel if i == 0 else out_channels

            res_blks.append(
                ResBlk(
                    in_channels=res_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    t_emb_dim=t_emb_dim,
                    down=False,
                    up=False,
                    n_groups=n_groups,
                    eps=eps,
                )
            )

        self.res_blks = nn.ModuleList(res_blks)

        if add_us:
            self.upsample = Upsample(
                in_channels=out_channels,
                out_channels=out_channels,
                use_conv=True,
            )
        else:
            self.upsample = None

    def forward(self, x: Tensor, z: List[Tensor], t: Tensor) -> Tensor:
        for res_blk in self.res_blks:
            _z = z[-1]
            z = z[:-1]
            x = torch.cat(tensors=[x, _z], dim=1)
            x = res_blk(x, t)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class AttnUpBlk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channel: int,
        out_channels: int,
        t_emb_dim: int,
        n_blks: int,
        n_groups: int,
        eps: float,
        add_us: bool,
        n_heads: int,
    ) -> None:
        super().__init__()

        res_blks = []
        attn_blks = []

        for i in range(n_blks):
            res_skip_channels = in_channels if (i == n_blks - 1) else out_channels
            res_in_channels = prev_out_channel if i == 0 else out_channels

            res_blks.append(
                ResBlk(
                    in_channels=res_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    t_emb_dim=t_emb_dim,
                    down=False,
                    up=False,
                    n_groups=n_groups,
                    eps=eps,
                )
            )
            attn_blks.append(
                MultiHeadAttention(
                    d_emb=out_channels,
                    n_heads=n_heads,
                )
            )

        self.norm = nn.GroupNorm(
            num_groups=n_groups,
            num_channels=out_channels,
            eps=eps,
            affine=True,
        )

        self.attn_blks = nn.ModuleList(attn_blks)
        self.res_blks = nn.ModuleList(res_blks)

        if add_us:
            self.upsample = Upsample(
                in_channels=out_channels,
                out_channels=out_channels,
                use_conv=True,
            )
        else:
            self.upsample = None

        return
    
    def forward(self, x: Tensor, z: List[Tensor], t: Tensor) -> Tensor:
        for res_blk, attn in zip(self.res_blks, self.attn_blks):
            _z = z[-1]
            z = z[:-1]
            x = torch.cat(tensors=[x, _z], dim=1)

            x = res_blk(x, t)

            x = self.norm(x)
            n, c, h, w = x.shape
            # [N, C, H, W] -> [N, C, HW] -> [N, HW, C]
            x = x.view(n, c, h * w).transpose(dim0=-1, dim1=-2)
            # [N, HW, C] -> [N, HW, C]
            x = attn(q=x, k=x, v=x)
            # [N, HW, C] -> [N, C, HW] -> [N, C, H, W]
            x = x.transpose(dim0=-1, dim1=-2).view(n, c, h, w)

        if self.upsample is not None:
            x = self.upsample(x)

        return x
