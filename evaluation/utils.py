from pathlib import Path
from typing import Callable, cast

import numpy as np
from torch import Tensor
import torch

from lgtm.model.motion_diffusion import Batch_T2M_EvaluationInfo, BatchData


def get_motion(motion_path: Path) -> Tensor | None:
    return torch.from_numpy(np.load(motion_path).copy()) if motion_path.exists() else None


def save_motion(motion_path: Path, motion: Tensor):
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(motion_path, motion.detach().cpu().numpy())


def motion_cache(cache_dir: Path, device: str):
    def wrapper(func: Callable[[BatchData], list[Tensor]]):
        def inner_wrapper(batch: BatchData):
            motions = list(map(get_motion, [cache_dir / f"{seg_id}.npy" for seg_id in batch.seg_ids]))

            for index in range(len(motions)):
                if motions[index] is not None:
                    motions[index] = cast(Tensor, motions[index]).to(device)

            filter_mask = [motion is None for motion in motions]
            if any(filter_mask):
                filtered_batch = BatchData(
                    seg_ids=[seg_id for seg_id, mask in zip(batch.seg_ids, filter_mask) if mask],
                    motions={
                        part_name: part_motions[filter_mask]
                        for part_name, part_motions in batch.motions.items()
                    },
                    body_part_texts={
                        part_name: [text for text, mask in zip(part_texts, filter_mask) if mask]
                        for part_name, part_texts in batch.body_part_texts.items()
                    },
                    frame_mask=batch.frame_mask[filter_mask],
                    whole_body_texts=[text for text, mask in zip(batch.whole_body_texts, filter_mask) if mask],
                    t2m_evaluation_info=Batch_T2M_EvaluationInfo(
                        batch.t2m_evaluation_info.pos_one_hots[filter_mask],
                        batch.t2m_evaluation_info.word_embeddings[filter_mask],
                        batch.t2m_evaluation_info.text_lengths[filter_mask],
                    ),
                )
                new_motions = func(filtered_batch)
                for seg_id, new_motion in zip(filtered_batch.seg_ids, new_motions):
                    save_motion(cache_dir / f"{seg_id}.npy", new_motion)
                for new_motion in new_motions:
                    motions[motions.index(None)] = new_motion

            return cast(list[Tensor], motions)

        return inner_wrapper

    return wrapper
