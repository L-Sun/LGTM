from typing import Sequence, TypeVar

import torch.utils.data

_T = TypeVar("_T")


class SequenceDataset(torch.utils.data.Dataset[_T], Sequence[_T]):
    pass
