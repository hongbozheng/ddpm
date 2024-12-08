from torch import Tensor

import torch
import torch.nn as nn
from attention import MultiHeadAttention
from bottleneck import Bottleneck
from downsample import Downsample
from resblk import ResBlk
from upsample import Upsample


class ResAttnBlk(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_groups: int,
            t_emb: int,
            d_emb: int,
            n_heads: int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.resblk = ResBlk(
            in_channels=in_channels,
            out_channels=out_channels,
            n_groups=n_groups,
            t_emb=t_emb,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.groupnorm = nn.GroupNorm(
            num_groups=n_groups,
            num_channels=out_channels,
        )
        self.self_attn = MultiHeadAttention(
            d_emb=d_emb,
            n_heads=n_heads,
            dropout=dropout,
        )

        return

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # [N, HW, C] -> [N, HW, C]
        x = self.resblk(x, t)
        x = self.dropout(x)

        _x = x
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        # [N, C, H, W] -> [N, C, HW] -> [N, HW, C]
        x = x.view(n, c, h * w).transpose(dim0=-1, dim1=-2)
        # [N, HW, C] -> [N, HW, C]
        x = self.self_attn(x, x, x)
        # [N, HW, C] -> [N, C, HW] -> [N, C, H, W]
        x = x.transpose(dim0=-1, dim1=-2).view(n, c, h, w)
        x += _x

        return x


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            channels: int,
            out_channels: int,
            n_groups: int,
            dropout: float,
            n_layers: int,
            n_heads: int,
    ) -> None:
        super().__init__()

        self.init_conv = self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.init_resblk = ResAttnBlk(
            in_channels=channels,
            out_channels=channels,
            n_groups=n_groups,
            t_emb=channels * 4,
            d_emb=channels,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.encoder = nn.ModuleList([])
        self.downsample = nn.ModuleList([])
        for i in range(0, n_layers):
            downsample = Downsample(
                in_channels=channels * 2 ** i,
                out_channels=channels * 2 ** i,
            )
            self.downsample.append(downsample)
            res_attn_blk = ResAttnBlk(
                in_channels=channels * 2 ** i,
                out_channels=channels * 2 ** (i + 1),
                n_groups=n_groups,
                t_emb=channels * 4,
                d_emb=channels * 2 ** (i + 1),
                n_heads=n_heads,
                dropout=dropout,
            )
            self.encoder.append(res_attn_blk)

        self.ds = Downsample(
            in_channels=channels * 2 ** n_layers,
            out_channels=channels * 2 ** n_layers,
        )
        self.bottleneck = Bottleneck(
            in_channels=channels * 2 ** n_layers,
            out_channels=channels * 2 ** (n_layers + 1),
            n_groups=n_groups,
            t_emb=channels * 4,
            d_emb=channels * 2 ** (n_layers + 1),
            n_heads=n_heads,
            dropout=dropout,
        )
        self.us = Upsample(
            in_channels=channels * 2 ** (n_layers + 1),
            out_channels=channels * 2 ** n_layers,
        )

        self.decoder = nn.ModuleList([])
        self.upsample = nn.ModuleList([])
        for i in range(n_layers, 0, -1):
            res_attn_blk = ResAttnBlk(
                in_channels=channels * 2 ** (i + 1),
                out_channels=channels * 2 ** i,
                n_groups=n_groups,
                t_emb=channels * 4,
                d_emb=channels * 2 ** i,
                n_heads=n_heads,
                dropout=dropout,
            )
            self.decoder.append(res_attn_blk)
            upsample = Upsample(
                in_channels=channels * 2 ** i,
                out_channels=channels * 2 ** (i - 1),
            )
            self.upsample.append(upsample)

        self.out_resblk = ResBlk(
            in_channels=channels,
            out_channels=channels,
            n_groups=n_groups,
            t_emb=channels * 4,
        )

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=n_groups, num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        return

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = t.unsqueeze(-1).type(torch.float).expand(-1, 128)
        x = self.init_conv(x)
        x = self.init_resblk(x, t)

        tmp = []

        for downsample, encoder in zip(self.downsample, self.encoder):
            x = downsample(x)
            x = encoder(x, t)
            tmp.append(x)

        x = self.ds(x)
        x = self.bottleneck(x, t)
        x = self.us(x)

        for decoder, upsample in zip(self.decoder, self.upsample):
            x = torch.cat(tensors=[x, tmp.pop()], dim=1)
            x = decoder(x, t)
            x = upsample(x)

        x = self.out_resblk(x, t)
        x = self.out(x)

        return x


model = UNet(
    in_channels=1,
    channels=32,
    out_channels=1,
    n_groups=32,
    dropout=0.2,
    n_layers=3,
    n_heads=4,
)
print(sum([p.numel() for p in model.parameters()]))
x = torch.randn(3, 1, 64, 64)
t = x.new_tensor([500] * x.shape[0])
# y = x.new_tensor([1] * x.shape[0])
print(model(x, t).shape)