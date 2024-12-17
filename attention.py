from typing import Optional, Tuple
from torch import Tensor

import config
import logger
import math
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_emb: int, n_heads: int) -> None:
        """
        Args:
            d_emb: embedding dimension
            n_heads: number of heads
        """
        super().__init__()
        self.d_emb = d_emb
        self.n_heads = n_heads
        assert self.d_emb % self.n_heads == 0, (
            logger.log_error(
                f"{self.emb_dim} is not divisible by {self.n_heads}"
            )
        )

        self.d_head = d_emb // self.n_heads
        self.w_q = nn.Linear(
            in_features=d_emb,
            out_features=d_emb,
            bias=False,
        )
        self.w_k = nn.Linear(
            in_features=d_emb,
            out_features=d_emb,
            bias=False,
        )
        self.w_v = nn.Linear(
            in_features=d_emb,
            out_features=d_emb,
            bias=False,
        )

        self.w_o = nn.Linear(
            in_features=d_emb,
            out_features=d_emb,
            bias=False,
        )

        return

    @staticmethod
    def attention(
            q: Tensor,
            k: Tensor,
            v: Tensor,
            mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q: query   [batch, n_heads, seq_len, head_dim]
            k: key     [batch, n_heads, seq_len, head_dim]
            v: value   [batch, n_heads, seq_len, head_dim]
            mask: mask [batch, seq_len]
            dropout: dropout probability
        Returns:
            [batch, n_heads, seq_len, head_dim],
            [batch, n_heads, seq_len, seq_len]
        """
        d_head = q.size(dim=-1)

        # [N, H, L, D_head] @ [N, H, D_head, L] -> [N, H, L, L]
        attn_scores = (q @ k.transpose(dim0=-2, dim1=-1)) / math.sqrt(d_head)

        if mask is not None:
            attn_scores.masked_fill_(mask=mask == 0, value=float('-inf'))

        attn_scores = attn_scores.softmax(dim=-1)

        # [N, H, L, D_head]
        return (attn_scores @ v), attn_scores

    def forward(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [N, L, D_emb] @ [D_emb, D_emb] -> [N, L, D_emb]
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # [N, L, D_emb] -> [N, L, H, D_emb] -> [N, H, L, D_emb]
        q = q.view(
            q.size(dim=0),
            q.size(dim=1),
            self.n_heads,
            self.d_head,
        ).transpose(dim0=1, dim1=2)
        k = k.view(
            k.size(dim=0),
            k.size(dim=1),
            self.n_heads,
            self.d_head,
        ).transpose(dim0=1, dim1=2)
        v = v.view(
            v.size(dim=0),
            v.size(dim=1),
            self.n_heads,
            self.d_head,
        ).transpose(dim0=1, dim1=2)

        x, self.attn_scores = MultiHeadAttention.attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
        )

        # [N, H, L, D_head] -> [N, L, H, D_head]
        x = x.transpose(dim0=1, dim1=2)
        # [N, L, H, D_head] -> [N, L, D_emb]
        x = x.contiguous().view(x.size(dim=0), -1, self.n_heads * self.d_head)
        # [N, L, D_emb] @ [D_emb, D_emb] -> [N, L, D_emb]
        x = self.w_o(x)

        return x
