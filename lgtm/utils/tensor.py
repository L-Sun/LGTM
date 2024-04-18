from typing import TypeVar

import torch
from torch import Tensor, nn


class Permute(nn.Module):
    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor):
        return torch.permute(x, self.dims)


_T = TypeVar("_T", bound=nn.Module)


def freeze_module(module: _T) -> _T:
    for param in module.parameters():
        param.requires_grad = False

    module.eval()
    return module
