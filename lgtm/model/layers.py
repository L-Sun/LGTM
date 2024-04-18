from typing import Literal
import torch
import torch.nn as nn
from torch import Tensor

ActivationFn = Literal["relu", "tanh", "gelu", "sigmoid", "silu", "none"]


def get_activation_fn(active_fn: ActivationFn) -> nn.Module:
    if active_fn == "relu":
        return nn.ReLU()
    elif active_fn == "tanh":
        return nn.Tanh()
    elif active_fn == "gelu":
        return nn.GELU()
    elif active_fn == "sigmoid":
        return nn.Sigmoid()
    elif active_fn == "silu":
        return nn.SiLU()
    elif active_fn == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function: {active_fn}")


class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_length=5000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # (1, max_length, d_model)
        self.pe: Tensor
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # (B, L, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        outputs, gate = x.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class ResidualConnection(nn.Module):
    def __init__(self, module: nn.Module, input_scale: float = 1.0, module_scale: float = 1.0):
        super().__init__()
        self.module = module
        self.input_scale = input_scale
        self.module_scale = module_scale

    def forward(self, x: Tensor) -> Tensor:
        return self.input_scale * x + self.module_scale * self.module(x)
