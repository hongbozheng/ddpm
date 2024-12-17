from torch import Tensor
from typing import Sequence

import math
import torch
import torch.nn as nn
from bottleneck import Bottleneck
from downblk import AttnDownBlk, DownBlk
from upblk import AttnUpBlk, UpBlk


class PositionalEncoding(nn.Module):
    def __init__(self, channels) -> None:
        """
        Initializes the PositionalEncoding layer.
        :param channels: Number of channels (embedding size).
        :param device: Device for tensor computation (e.g., 'cpu' or 'cuda').
        """
        super().__init__()
        self.channels = channels

        # Precompute the inverse frequency for positional encodings
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.register_buffer(name="inv_freq", tensor=inv_freq, persistent=False)

        return

    def forward(self, t: Tensor) -> Tensor:
        """
        Compute the positional encoding for input tensor t.
        :param t: Input tensor of shape (batch_size, 1).
        :return: Positional encoding tensor of shape (batch_size, channels).
        """
        # Expand t and compute sine and cosine encodings
        pos_enc_a = torch.sin(t.repeat(1, self.channels // 2) * self.inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.channels // 2) * self.inv_freq)
        # Concatenate sine and cosine encodings
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)

        return pos_enc

def get_timestep_embedding(
        timesteps: Tensor,
        embedding_dim: int,
        max_period: int = 10000,
) -> Tensor:
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding


def get_down_blk(
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        n_blks: int,
        n_groups: int,
        eps: float,
        add_ds: bool,
        with_attn: bool,
        n_heads: int,
) -> nn.Module:
    if with_attn:
        return AttnDownBlk(
            in_channels=in_channels,
            out_channels=out_channels,
            t_emb_dim=t_emb_dim,
            n_blks=n_blks,
            n_groups=n_groups,
            eps=eps,
            add_ds=add_ds,
            n_heads=n_heads,
        )
    else:
        return DownBlk(
            in_channels=in_channels,
            out_channels=out_channels,
            t_emb_dim=t_emb_dim,
            n_blks=n_blks,
            n_groups=n_groups,
            eps=eps,
            add_ds=add_ds,
        )


def get_bottleneck(
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        n_groups: int,
        eps: float,
        n_heads: int,
) -> nn.Module:
    return Bottleneck(
        in_channels=in_channels,
        out_channels=out_channels,
        t_emb_dim=t_emb_dim,
        n_groups=n_groups,
        eps=eps,
        n_heads=n_heads,
    )


def get_up_blk(
        in_channels: int,
        prev_out_channel: int,
        out_channels: int,
        t_emb_dim: int,
        n_blks: int,
        n_groups: int,
        eps: float,
        add_us: bool,
        with_attn: bool,
        n_heads: int,
) -> nn.Module:
    if with_attn:
        return AttnUpBlk(
            in_channels=in_channels,
            prev_out_channel=prev_out_channel,
            out_channels=out_channels,
            t_emb_dim=t_emb_dim,
            n_blks=n_blks,
            n_groups=n_groups,
            eps=eps,
            add_us=add_us,
            n_heads=n_heads,
        )
    else:
        return UpBlk(
            in_channels=in_channels,
            prev_out_channel=prev_out_channel,
            out_channels=out_channels,
            t_emb_dim=t_emb_dim,
            n_blks=n_blks,
            n_groups=n_groups,
            eps=eps,
            add_us=add_us,
        )


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_channels: Sequence[int],
            n_blks: Sequence[int],
            attn: Sequence[bool],
            n_groups: int,
            eps: float,
            n_heads: int,
            t_emb_dim: int,
            n_classes: int,
    ) -> None:
        super().__init__()

        self.init_conv = self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )


        # self.pe = PositionalEncoding(channels=t_emb_dim)
        self.label_emb = nn.Embedding(
            num_embeddings=n_classes,
            embedding_dim=t_emb_dim,
        )

        self.t_emb = nn.Sequential(
            nn.Linear(in_features=n_channels[0], out_features=t_emb_dim),
            nn.SiLU(),
            nn.Linear(in_features=t_emb_dim, out_features=t_emb_dim),
        )

        self.c_emb = nn.Embedding(
            num_embeddings=n_classes,
            embedding_dim=t_emb_dim,
        )

        self.down_blks = nn.ModuleList([])
        oc = n_channels[0]
        for i in range(len(n_channels)):
            ic = oc
            oc = n_channels[i]
            final_blk = i == len(n_channels) - 1

            down_blk = get_down_blk(
                in_channels=ic,
                out_channels=oc,
                t_emb_dim=t_emb_dim,
                n_blks=n_blks[i],
                n_groups=n_groups,
                eps=eps,
                add_ds=not final_blk,
                with_attn=attn[i],
                n_heads=n_heads,
            )

            self.down_blks.append(down_blk)

        self.bottleneck = get_bottleneck(
            in_channels=n_channels[-1],
            out_channels=n_channels[-1],
            t_emb_dim=t_emb_dim,
            n_groups=n_groups,
            eps=eps,
            n_heads=n_heads,
        )

        self.up_blks = nn.ModuleList([])
        n_channels = list(reversed(n_channels))
        n_blks = list(reversed(n_blks))
        attn = list(reversed(attn))
        oc = n_channels[0]
        for i in range(len(n_channels)):
            prev_oc = oc
            oc = n_channels[i]
            ic = n_channels[min(i + 1, len(n_channels) - 1)]
            final_blk = i == len(n_channels) - 1

            up_blk = get_up_blk(
                in_channels=ic,
                prev_out_channel=prev_oc,
                out_channels=oc,
                t_emb_dim=t_emb_dim,
                n_blks=n_blks[i] + 1,
                n_groups=n_groups,
                eps=eps,
                add_us=not final_blk,
                with_attn=attn[i],
                n_heads=n_heads,
            )

            self.up_blks.append(up_blk)

        self.out = nn.Sequential(
            nn.GroupNorm(
                num_groups=n_groups,
                num_channels=n_channels[-1],
                eps=eps,
                affine=True,
            ),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=n_channels[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self._reset_parameters()

        return

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(tensor=p)

        return

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        t = get_timestep_embedding(timesteps=t, embedding_dim=32)
        t = t.to(dtype=x.dtype)
        t = self.t_emb(t)

        # t = t.unsqueeze(dim=-1).to(dtype=torch.float32)
        # t = self.pe(t)

        if y is not None:
            t += self.c_emb(y).to(dtype=x.dtype)

        x = self.init_conv(x)
        # print("init conv", x.shape)

        z = [x]

        for down_blk in self.down_blks:
            x, res = down_blk(x, t)
            # print("down blk", x.shape)
            for r in res:
                z.append(r)

        x = self.bottleneck(x, t)
        # print("after bottleneck", x.shape)

        for up_blk in self.up_blks:
            res = z[-len(up_blk.res_blks):]
            z = z[:-len(up_blk.res_blks)]
            x = up_blk(x, res, t)
            # print("up blk", x.shape)

        x = self.out(x)

        return x
