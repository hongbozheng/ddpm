from torch import Tensor

from attention import MultiHeadAttention
from resblk import ResBlk
from torch import nn


class Bottleneck(nn.Module):
    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            t_emb_dim:int,
            n_groups: int,
            eps: float,
            n_heads:int,
    ) -> None:
        super().__init__()
        self.res_blk_0 = ResBlk(
            in_channels=in_channels,
            out_channels=in_channels,
            t_emb_dim=t_emb_dim,
            down=False,
            up=False,
            n_groups=n_groups,
            eps=eps,
        )
        self.groupnorm = nn.GroupNorm(
            num_groups=n_groups,
            num_channels=out_channels,
            eps=eps,
        )
        self.self_attn = MultiHeadAttention(
            d_emb=in_channels,
            n_heads=n_heads,
        )
        self.res_blk_1 = ResBlk(
            in_channels=in_channels,
            out_channels=in_channels,
            t_emb_dim=t_emb_dim,
            down=False,
            up=False,
            n_groups=n_groups,
            eps=eps,
        )

        return

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # [N, C, H, W] -> [N, C, H, W]
        x = self.res_blk_0(x, t)

        x = self.groupnorm(x)
        n, c, h, w = x.shape
        # [N, C, H, W] -> [N, C, HW] -> [N, HW, C]
        x = x.view(n, c, h * w).transpose(dim0=-1, dim1=-2)
        # [N, HW, C] -> [N, HW, C]
        x = self.self_attn(q=x, k=x, v=x)
        # [N, HW, C] -> [N, C, HW] -> [N, C, H, W]
        x = x.transpose(dim0=-1, dim1=-2).view(n, c, h, w)

        # [N, C, H, W] -> [N, C, H, W]
        x = self.res_blk_1(x, t)

        return x
