import numpy as np
from torch import Tensor
import torch


def resample_motion_features(features: np.ndarray, origin_fps: float, new_fps: float) -> np.ndarray:
    """Resample motion signal features along the first axis"""

    # resample using griddata
    from scipy.interpolate import griddata

    origin_frame_time, new_frame_time = 1.0 / origin_fps, 1.0 / new_fps
    origin_shape = features.shape
    origin_num_frames = features.shape[0]
    duration = origin_num_frames * origin_frame_time
    new_num_frames = int(np.ceil(duration * new_fps))

    if origin_num_frames == new_num_frames:
        return features

    assert origin_num_frames >= 4, f"origin_num_frames must be greater than 4, but got {origin_num_frames}"

    original_times = np.linspace(0, origin_frame_time, origin_num_frames, dtype=np.float32)
    new_times = np.linspace(0, original_times[-1], new_num_frames, dtype=np.float32)

    resample_features = griddata(original_times, features.reshape([origin_num_frames, -1]), new_times, method='cubic').reshape(new_num_frames, *origin_shape[1:]).astype(np.float32)

    return resample_features


def pad_random_truncate_sequences_np(sequences: list[np.ndarray], padding_value=0.0, length: int | None = None):
    """pad all sequence to the same length with given value, and truncate randomly

    Args:
        sequences: list of variable length
        padding_value: value for padded elements.
        length: if None, the max length of sequences will be used, else use given length (may truncate sequences)

    Returns:
        tuple[np.ndarray,np.ndarray]: padded sequences and mask
    """

    if length is None:
        length = max([len(seq) for seq in sequences])

    padded_sequences = np.full([len(sequences), length, *sequences[0].shape[1:]], padding_value, dtype=sequences[0].dtype)
    mask = np.full([len(sequences), length], False)

    for i, seq in enumerate(sequences):
        copy_length = min(len(seq), length)
        start = np.random.randint(0, len(seq) - copy_length + 1)
        padded_sequences[i, :copy_length] = seq[start:start + copy_length]
        mask[i, :copy_length] = True

    return padded_sequences, mask


def pad_random_truncate_sequences(sequences: list[Tensor], padding_value=0.0, length: int | None = None) -> tuple[Tensor, Tensor]:
    if length is None:
        length = max([len(seq) for seq in sequences])

    device = sequences[0].device
    dtype = sequences[0].dtype

    padded_sequences = torch.full([len(sequences), length, *sequences[0].shape[1:]], padding_value, dtype=dtype, device=device)
    mask = torch.full([len(sequences), length], False)

    for i, seq in enumerate(sequences):
        copy_length = min(len(seq), length)
        start = np.random.randint(0, len(seq) - copy_length + 1)
        padded_sequences[i, :copy_length] = seq[start:start + copy_length]
        mask[i, :copy_length] = True

    return padded_sequences, mask
