from torch import Tensor
from typing import List

import torch.nn as nn
from attention import MultiHeadAttention
from downsample import Downsample
from resblk import ResBlk


class DownBlk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        n_blks: int,
        n_groups: int,
        eps: float,
        add_ds: bool,
    ) -> None:
        super().__init__()

        res_blks = []

        for i in range(n_blks):
            in_channels = in_channels if i == 0 else out_channels
            res_blks.append(
                ResBlk(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    t_emb_dim=t_emb_dim,
                    down=False,
                    up=False,
                    n_groups=n_groups,
                    eps=eps,
                )
            )
        
        self.res_blks = nn.ModuleList(res_blks)

        if add_ds:
            self.downsample = Downsample(
                in_channels=out_channels,
                out_channels=out_channels,
                use_conv=True,
            )
        else:
            self.downsample = None

        return

    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, list[Tensor]]:
        z = []

        for res_blk in self.res_blks:
            x = res_blk(x, t)
            z.append(x)

        if self.downsample is not None:
            x = self.downsample(x)
            z.append(x)

        return x, z


class AttnDownBlk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        n_blks: int,
        n_groups: int,
        eps: float,
        add_ds: bool,
        n_heads: int,
    ) -> None:
        super().__init__()

        res_blks = []
        attn_blks = []

        for i in range(n_blks):
            in_channels = in_channels if i == 0 else out_channels
            res_blks.append(
                ResBlk(
                    in_channels=in_channels,
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

        if add_ds:
            self.downsample = Downsample(
                in_channels=out_channels,
                out_channels=out_channels,
                use_conv=True,
            )
        else:
            self.downsample = None

        return
    
    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, List[Tensor]]:
        z = []
        for res_blk, attn in zip(self.res_blks, self.attn_blks):
            x = res_blk(x, t)

            x = self.norm(x)
            n, c, h, w = x.shape
            # [N, C, H, W] -> [N, C, HW] -> [N, HW, C]
            x = x.view(n, c, h * w).transpose(dim0=-1, dim1=-2)
            # [N, HW, C] -> [N, HW, C]
            x = attn(q=x, k=x, v=x)
            # [N, HW, C] -> [N, C, HW] -> [N, C, H, W]
            x = x.transpose(dim0=-1, dim1=-2).view(n, c, h, w)

            z.append(x)

        if self.downsample is not None:
            x = self.downsample(x)
            z.append(x)

        return x, z
