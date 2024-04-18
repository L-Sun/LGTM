import copy

import torch.nn as nn
from torch import Tensor

from lgtm.model.layers import GLU, PositionEncoding, ResidualConnection
from lgtm.utils.tensor import Permute


class ConformerAttentionModule(nn.Module):
    def __init__(self, input_length: int, input_dim: int, num_heads: int = 1, dropout=0.1) -> None:
        super().__init__()

        self.position_encoding = PositionEncoding(input_dim, 1000)

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.token_self_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout_1 = nn.Dropout(dropout)

        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.emb_self_attn = nn.MultiheadAttention(
            embed_dim=input_length,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_key_padding_mask: Tensor | None = None) -> Tensor:
        x: Tensor = src
        x = x + self._token_self_attn_block(x, src_key_padding_mask)
        x = x + self._emb_self_attn_block(x)
        return x

    def _token_self_attn_block(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.position_encoding(self.layer_norm_1(x))
        return self.dropout_1(self.token_self_attn(x, x, x, key_padding_mask=mask)[0])

    def _emb_self_attn_block(self, x: Tensor) -> Tensor:
        x = self.position_encoding(self.layer_norm_2(x))
        x = x.permute(0, 2, 1)
        x = self.dropout_2(self.emb_self_attn(x, x, x)[0])
        x = x.permute(0, 2, 1)
        return self.dropout_2(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        input_length: int,
        input_dim: int,
        depth_kernel_size: int,
        feed_forward_expansion: int = 4,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.feed_forward = ResidualConnection(nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, feed_forward_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_expansion, input_dim),
            nn.Dropout(dropout),
        ), module_scale=0.5)

        self.attn = ConformerAttentionModule(input_length, input_dim, num_heads, dropout)

        self.conv_module = ResidualConnection(nn.Sequential(
            nn.LayerNorm(input_dim),
            Permute(0, 2, 1),
            nn.Conv1d(input_dim, 2 * input_dim, kernel_size=1),
            GLU(dim=1),
            nn.Conv1d(input_dim, input_dim, kernel_size=depth_kernel_size, padding=(depth_kernel_size - 1) // 2, groups=input_dim),
            nn.BatchNorm1d(input_dim),
            nn.SiLU(),
            nn.Conv1d(input_dim, input_dim, kernel_size=1),
            nn.Dropout(dropout),
            Permute(0, 2, 1),
        ))

        self.feed_forward = ResidualConnection(nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, feed_forward_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_expansion, input_dim),
            nn.Dropout(dropout),
        ), module_scale=0.5)

        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x: Tensor = src
        x = self.feed_forward(x)
        x = self.attn(x, src_key_padding_mask)
        x = self.conv_module(x)
        x = self.feed_forward(x)
        x = self.layer_norm(x)
        return x


class ConformerEncoder(nn.Module):
    def __init__(self, conformer_block: ConformerBlock, num_blocks: int):
        super().__init__()
        self.conformer_blocks = nn.ModuleList([copy.deepcopy(conformer_block) for _ in range(num_blocks)])

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:

        x: Tensor = src
        for block in self.conformer_blocks:
            x = block(x, src_key_padding_mask)

        return x
