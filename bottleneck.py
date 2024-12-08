from torch import Tensor

from torch import nn
from attention import MultiHeadAttention
from resblk import ResBlk
from upsample import Upsample


class Bottleneck(nn.Module):
    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            n_groups: int,
            t_emb:int,
            d_emb:int,
            n_heads:int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.res_blk = ResBlk(
            in_channels=in_channels,
            out_channels=out_channels,
            n_groups=n_groups,
            t_emb=t_emb,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.groupnorm = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)
        self.self_attn = MultiHeadAttention(
            d_emb=d_emb,
            n_heads=n_heads,
            dropout=dropout,
        )

        return

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # [N, HW, C] -> [N, HW, C]
        x = self.res_blk(x, t)
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
