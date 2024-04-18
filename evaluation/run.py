import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, cast

import pandas as pd
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl

pl.seed_everything(1234)

from evaluation.utils import motion_cache
from lgtm.dataset.HumanML3D import (BodyPart_HumanML3D, rearrange_humanml3d_features, recover_rearranged_humanml3d_features)
from lgtm.metrics import (Diversity, FrechetInceptionDistance, RetrievalPrecision, Similarity)
from lgtm.model.motion_diffusion import BatchData, MotionDiffusion_DataModule
from lgtm.model.TMR import TMR_Wrapper
from lgtm.utils.data_processing import pad_random_truncate_sequences
from lgtm.utils.tensor import freeze_module


def get_group_start_end_indices(arr_length: int, num_groups: int):
    group_size = arr_length // num_groups
    remainder = arr_length % num_groups
    start_indices = [0]
    start_index = 0

    for i in range(1, num_groups):
        start_index += group_size + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0
        start_indices.append(start_index)
    start_indices.append(arr_length)

    return list(zip(start_indices[:-1], start_indices[1:]))


LatentComputeFunction = Callable[[list[Tensor], list[Tensor], list[str], BatchData], tuple[Tensor, Tensor, Tensor]]
SampleGenerator = Callable[[BatchData], list[Tensor]]


class MetricCalculator:
    def __init__(
        self,
        name: str,
        dataloader: DataLoader[BatchData],
        generator: SampleGenerator,
        latent_compute_functions: dict[str, LatentComputeFunction],
        motion_latent_dims: dict[str, int],
        device: str,
        split_index: int | None,
    ):
        self.name = name
        self.dataloader = dataloader
        self.device = device

        self.generator = generator
        self.latent_compute_functions = latent_compute_functions

        part_names = list(motion_latent_dims.keys())

        self.diversity = Diversity(motion_latent_dims["whole_body"]).to(self.device)
        self.fid = FrechetInceptionDistance(motion_latent_dims["whole_body"]).to(self.device)
        self.retrieval_precision = RetrievalPrecision(top_k=3).to(self.device)

        self.similarities = {                       #
            part_name: Similarity().to(self.device)
            for part_name in part_names
        }

        if split_index is not None:
            self.batch_start, self.batch_end = get_group_start_end_indices(len(dataloader), torch.cuda.device_count())[split_index]
        else:
            self.batch_start, self.batch_end = 0, len(dataloader)

    def compute(self) -> dict[str, float]:
        self._batch_loop()

        results = dict[str, float]()
        results |= self.diversity.compute()
        results |= self.fid.compute()
        results |= self.retrieval_precision.compute()

        for part_name, similarity in self.similarities.items():
            results |= {f"{part_name}_{key}": value for key, value in similarity.compute().items()}

        return results

    def _get_motion_dict(self, motions: list[Tensor]):
        temps = [rearrange_humanml3d_features(motion) for motion in motions]
        body_part_motions = {part_name: [motion[part_name] for motion in temps] for part_name in BodyPart_HumanML3D.body_part_feature_dims.keys()}
        return {
            "whole_body": motions,
            **body_part_motions,
        }

    def _batch_loop(self):
        progress_bar = tqdm(total=self.batch_end - self.batch_start, desc=f"Loop for {self.name}")
        index = 0
        with torch.no_grad():
            for batch in self.dataloader:
                if index < self.batch_start:
                    index += 1
                    continue
                if index >= self.batch_end:
                    break

                index += 1
                progress_bar.update(1)

                batch = cast(BatchData, batch)
                part_names = list(batch.motions.keys())

                motions_gt = self._get_motion_dict(get_ground_generator(self.device)(batch))
                motions = self._get_motion_dict(self.generator(batch))
                texts = {
                    "whole_body": batch.whole_body_texts,
                    **batch.body_part_texts,
                }

                for name in motions.keys():

                    latent_motion, latent_motion_gt, latent_text_gt = self.latent_compute_functions[name](
                        motions[name],
                        motions_gt[name],
                        texts[name],
                        batch,
                    )
                    if name == "whole_body":
                        self.diversity.update(latent_motion)
                        self.fid.update(latent_motion_gt, real=True)
                        self.fid.update(latent_motion, real=False)
                        self.retrieval_precision.update(latent_motion, latent_text_gt)

                    self.similarities[name].update(latent_motion, latent_text_gt)


def get_ground_generator(device: str):
    def ground_generator(batch: BatchData) -> list[Tensor]:
        whole_body_motions = recover_rearranged_humanml3d_features(batch.motions).to(device)

        lengths = batch.frame_mask.sum(dim=1).to(device)
        results = [motion[:length] for motion, length in zip(whole_body_motions, lengths)]
        return results

    return ground_generator


def get_lgtm_generator(lgtm_ckpt: Path, device: str) -> SampleGenerator:
    from lgtm.model.motion_diffusion import MotionDiffusion

    lgtm_model = MotionDiffusion.load_from_checkpoint(lgtm_ckpt, map_location=device)
    lgtm_model.freeze()

    version = Path(lgtm_ckpt).parts[-3]

    @motion_cache(Path(f"results/motions/lgtm/{version}"), device)
    def lgtm_generator(batch: BatchData):
        lengths = [int(mask.sum().item()) for mask in batch.frame_mask]
        body_part_motions = lgtm_model.sample(
            batch.whole_body_texts,
            batch.body_part_texts,
            lengths,
            num_inference_steps=1000,
            tqdm_kwargs={
                "position": 1,
                "leave": False,
                "desc": "Sample Motion",
            },
        )

        whole_body_motions = [recover_rearranged_humanml3d_features(part_motion) for part_motion in body_part_motions]
        return whole_body_motions

    return lgtm_generator


def get_mdm_generator(mdm_model_path: Path, device: str) -> SampleGenerator:
    from evaluation.mdm_generator import MDM_Generator
    model = freeze_module(MDM_Generator(mdm_model_path, progress=False)).to(device)

    @motion_cache(Path("results/motions"), device)
    def mdm_generator(batch: BatchData):
        lengths = [int(mask.sum().item()) for mask in batch.frame_mask]
        whole_body_motions = model.generate(batch.whole_body_texts, lengths)
        return whole_body_motions

    return mdm_generator


def get_mld_generator(mld_model_path: Path, device: str) -> SampleGenerator:
    from evaluation.mld_generator import MLD_Generator
    model = freeze_module(MLD_Generator(mld_model_path, progress=False)).to(device)

    @motion_cache(Path("results/motions/mld"), device)
    def mld_generator(batch: BatchData):
        lengths = [int(mask.sum().item()) for mask in batch.frame_mask]
        whole_body_motions = model.generate(batch.whole_body_texts, lengths, return_feat=True)
        return whole_body_motions

    return mld_generator


# FIXME: motion_diffuse_model_path not used
def get_motion_diffuse_generator(motion_diffuse_model_path: Path, device: str) -> SampleGenerator:
    from evaluation.motion_diffuse_generator import MotionDiffuse_Generator
    model = freeze_module(MotionDiffuse_Generator("checkpoints/t2m/t2m_motiondiffuse/opt.txt", device))

    @motion_cache(Path("results/motions/MotionDiffuse"), device)
    def motion_diffuse_generator(batch: BatchData):
        lengths = [int(mask.sum().item()) for mask in batch.frame_mask]
        whole_body_motions = model.generate(batch.whole_body_texts, lengths)
        return whole_body_motions

    return motion_diffuse_generator


def get_tmr_latent_compute_function(tmr_encoder: TMR_Wrapper) -> LatentComputeFunction:
    def tmr_latent_compute_function(motions: list[Tensor], motions_gt: list[Tensor], texts: list[str], batch: BatchData):

        latent_motions = tmr_encoder.encode_motion(motions)
        latent_motions_gt = tmr_encoder.encode_motion(motions_gt)
        latent_text = tmr_encoder.encode_text(texts)
        return latent_motions, latent_motions_gt, latent_text

    return tmr_latent_compute_function


def get_t2m_latent_compute_function(t2m_evaluator_dir: Path, device: str, mean: Tensor, std: Tensor):
    from evaluation.t2m_evaluator import T2M_Evaluator
    evaluator = freeze_module(T2M_Evaluator(t2m_evaluator_dir, "humanml")).to(device)

    def t2m_latent_compute_function(motions: list[Tensor], motions_gt: list[Tensor], texts: list[str], batch: BatchData):
        motions = [evaluator.normalize(motion * std + mean) for motion in motions]
        motions_gt = [evaluator.normalize(motion_gt * std + mean) for motion_gt in motions_gt]

        _motions, _masks = pad_random_truncate_sequences(motions)
        _motions_gt, _masks_gt = pad_random_truncate_sequences(motions_gt)

        lengths = _masks.sum(dim=1)
        lengths_gt = _masks_gt.sum(dim=1)

        latent_motions = evaluator.encode_motion(_motions, lengths)
        latent_motions_gt = evaluator.encode_motion(_motions_gt, lengths_gt)
        latent_texts = evaluator.encode_text(
            batch.t2m_evaluation_info.word_embeddings.to(device),
            batch.t2m_evaluation_info.pos_one_hots.to(device),
            batch.t2m_evaluation_info.text_lengths,
        )

        return latent_motions, latent_motions_gt, latent_texts

    return t2m_latent_compute_function


def main():
    parser = ArgumentParser()
    parser.add_argument("--tmr_whole_body_model_dir", type=Path, default=Path("third_packages/TMR/models/tmr_humanml3d_guoh3dfeats"))
    parser.add_argument("--tmr_body_part_model_dir", type=Path, default=Path("third_packages/TMR/models/body_part_tmr/"))
    parser.add_argument("--humanml3d_dir", type=Path, default=Path("third_packages/HumanML3D"))
    parser.add_argument("--glove_dir", type=Path, default=Path("data/glove"))
    parser.add_argument("--body_part_annotations_path", type=Path, default=Path("third_packages/TMR/datasets/annotations/humanml3d/body_part_annotations.json"))
    parser.add_argument("--t2m_evaluator", type=Path, default=Path("data/t2m_evaluator"))

    parser.add_argument("--t2m", action="store_true")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--split_index", type=int)

    parser.add_argument("--ground_truth", action="store_true")
    parser.add_argument("--mdm_model_path", type=Path)
    parser.add_argument("--lgtm_ckpt", type=Path)
    parser.add_argument("--mld_ckpt", type=Path)
    parser.add_argument("--motion_diffuse_ckpt", type=Path)

    parser.add_argument("--save_path", type=Path)

    args = parser.parse_args()
    device = f"cuda:{args.device}"
    save_path: Path = args.save_path
    batch_size: int = args.batch_size

    dataset = MotionDiffusion_DataModule(args.humanml3d_dir, args.glove_dir, args.body_part_annotations_path, batch_size=batch_size)
    dataset.setup("test")
    dataloader = dataset.test_dataloader()

    mean = torch.from_numpy(dataset.test_data.humanml3d.mean).to(device)
    std = torch.from_numpy(dataset.test_data.humanml3d.std).to(device)

    part_names = list(dataset.body_part_feature_dims.keys())

    tmr_encoders = {
        "whole_body": freeze_module(TMR_Wrapper(args.tmr_whole_body_model_dir)).to(device),
        **{
            part_name: freeze_module(TMR_Wrapper(args.tmr_body_part_model_dir / part_name)).to(device)
            for part_name in part_names
        },
    }

    latent_compute_functions: dict[str, LatentComputeFunction] = {
        "whole_body": get_tmr_latent_compute_function(tmr_encoders["whole_body"]),
        **{
            part_name: get_tmr_latent_compute_function(tmr_encoders[part_name])
            for part_name in part_names
        },
    }

    motion_latent_dims = {                                   #
        "whole_body": tmr_encoders["whole_body"].latent_dim,
        **{
            part_name: tmr_encoders[part_name].latent_dim
            for part_name in part_names
        },
    }

    if args.t2m:
        latent_compute_functions["whole_body"] = get_t2m_latent_compute_function(args.t2m_evaluator, device, mean, std)
        motion_latent_dims["whole_body"] = 512

    calculator: MetricCalculator | None = None

    num_selected = sum([args.ground_truth, args.mdm_model_path is not None, args.lgtm_ckpt is not None])
    if num_selected > 1:
        raise RuntimeError("Only one calculator can be selected")

    if args.split_index:
        logger.info("enable split, please make sure run 0->num_gpu_devices split tasks")

    if args.ground_truth:
        calculator = MetricCalculator(
            "ground_truth",
            dataloader,
            get_ground_generator(device),
            latent_compute_functions,
            motion_latent_dims,
            args.device,
            args.split_index,
        )

    if args.mdm_model_path is not None:
        calculator = MetricCalculator(
            "mdm",
            dataloader,
            get_mdm_generator(args.mdm_model_path, device),
            latent_compute_functions,
            motion_latent_dims,
            args.device,
            args.split_index,
        )

    if args.lgtm_ckpt is not None:
        calculator = MetricCalculator(
            "our",
            dataloader,
            get_lgtm_generator(args.lgtm_ckpt, device),
            latent_compute_functions,
            motion_latent_dims,
            args.device,
            args.split_index,
        )

    if args.mld_ckpt is not None:
        calculator = MetricCalculator(
            "mld",
            dataloader,
            get_mld_generator(args.mld_ckpt, device),
            latent_compute_functions,
            motion_latent_dims,
            args.device,
            args.split_index,
        )

    if args.motion_diffuse_ckpt is not None:
        calculator = MetricCalculator(
            "motion_diffuse",
            dataloader,
            get_motion_diffuse_generator(args.motion_diffuse_ckpt, device),
            latent_compute_functions,
            motion_latent_dims,
            args.device,
            args.split_index,
        )

    if calculator is None:
        raise RuntimeError("No calculator is selected")

    logger.info(f"Start computing {calculator.name} metrics")
    pd.DataFrame({
        "name": calculator.name,
        **calculator.compute(),
    }, index=[0]).to_excel(save_path, index=False)


if __name__ == "__main__":
    main()
