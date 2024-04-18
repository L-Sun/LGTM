import typing
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pytorch_lightning as pl
import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.schedulers.scheduling_ddim import (DDIMScheduler, DDIMSchedulerOutput)
from pytorch_lightning.cli import LightningArgumentParser
from torch import FloatTensor, IntTensor, Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from lgtm.model.conformer import ConformerBlock, ConformerEncoder
from lgtm.model.layers import ActivationFn, get_activation_fn
from lgtm.model.TMR import TMR_Wrapper
from lgtm.model.smooth_net import SmoothNet
# from lgtm.model.text_encoder import CLIP_TextEncoder
from lgtm.utils.tensor import freeze_module


class MotionDiffusion(pl.LightningModule):
    def __init__(
            self,
            body_part_feature_dims: dict[str, int],
            max_frames: int,
            latent_dim: int = 128,
            num_conformer_blocks: int = 4,
            num_attention_layers: int = 4,
            num_heads=4,
            num_time_steps: int = 1000,
            activation_fn: ActivationFn = "silu",
            dropout: float = 0.1,
            learning_rate=1e-4,
            tmr_dir: Path = Path("third_packages/TMR"),
    ):
        """Motion Diffusion

        Args:
            input_shape: shape of motion data `(max_frames, feature_dim)`.
            latent_dim: dimension of latent space.
            clip_model_name: name of CLIP model.
            num_time_steps: number of time steps during training.
        """

        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.max_frames = max_frames

        self.body_part_feature_dims = body_part_feature_dims
        self.body_part_names = list(body_part_feature_dims.keys())
        self.body_part_names.sort()

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_time_steps,
            prediction_type="sample",
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
        )

        self.tmr_encoders = freeze_module(nn.ModuleDict({
            "whole_body": TMR_Wrapper(tmr_dir / "models" / "tmr_humanml3d_guoh3dfeats"),
            **{
                part_name: TMR_Wrapper(tmr_dir / "models" / "body_part_tmr" / part_name)
                for part_name in self.body_part_names
            },
        }))

        self.time_embedding = nn.Sequential(OrderedDict([                                        # (B)
            ("time_proj", Timesteps(latent_dim, True, 0)),                                       # (B, Z)
            ("time_embedding", TimestepEmbedding(latent_dim, latent_dim, act_fn=activation_fn)), # (B, Z)
        ]))

        # clip_text_encoder = freeze_module(CLIP_TextEncoder())
        # self.body_part_text_encoders = nn.ModuleDict({               #
        #     part_name: nn.Sequential(
        #         clip_text_encoder,
        #         nn.Linear(clip_text_encoder.output_dim, latent_dim),
        #     )
        #     for part_name in self.body_part_names
        # })

        # self.whole_body_text_encoder = nn.Sequential(
        #     clip_text_encoder,
        #     nn.Linear(clip_text_encoder.output_dim, latent_dim),
        # )

        self.body_part_text_encoders = nn.ModuleDict({ #
            part_name: nn.Sequential(
                self.tmr_encoders[part_name],
                nn.Linear(256, latent_dim),
            )
            for part_name in self.body_part_names
        })

        self.whole_body_text_encoder = nn.Sequential(
            self.tmr_encoders["whole_body"],
            nn.Linear(256, latent_dim),
        )

        self.part_motion_encoders = nn.ModuleDict({      #
            part_name: PartMotionEncoder(
                self.body_part_text_encoders[part_name],
                self.max_frames,
                body_part_feature_dims[part_name],
                latent_dim,
                dropout,
                num_heads,
                num_conformer_blocks,
            )
            for part_name in self.body_part_names
        })

        self.whole_body_attention = WholeBodyAttention(
            self.whole_body_text_encoder,
            self.max_frames,
            latent_dim,
            body_part_feature_dims,
            num_heads,
            num_attention_layers,
            activation_fn,
            dropout,
        )

    def forward(self, x_t: dict[str, Tensor], t: IntTensor, whole_body_text: list[str], part_texts: dict[str, list[str]], frame_mask: Tensor | None = None):
        batch_size = t.shape[0]

        frame_mask = torch.ones(batch_size, self.max_frames, device=self.device, dtype=torch.bool) if frame_mask is None else frame_mask
        time_embeddings: Tensor = self.time_embedding(t)

        temp: dict[str, Tensor] = {                          #
            part_name: self.part_motion_encoders[part_name](
                x_t[part_name],
                part_texts[part_name],
                frame_mask,
                time_embeddings,
            )
            for part_name in self.body_part_names
        }

        output: dict[str, Tensor] = self.whole_body_attention(temp, whole_body_text, frame_mask)

        return output

    def sample(
        self,
        whole_texts: list[str],
        part_texts: dict[str, list[str]],
        lengths: list[int] | None = None,
        eta=0.0,
        num_inference_steps=1000,
        tqdm_kwargs: dict[str, Any] = {},
    ) -> list[dict[str, Tensor]]:
        batch_size = len(whole_texts) if whole_texts else 1
        lengths = lengths if lengths is not None else [self.max_frames] * batch_size

        x_t = {                                                                                                                                  #
            part_name: typing.cast(Tensor, torch.randn(batch_size, self.max_frames, self.body_part_feature_dims[part_name], device=self.device))
            for part_name in self.body_part_names
        }

        masks = torch.ones(batch_size, self.max_frames, dtype=torch.bool, device=self.device)
        for mask, length in zip(masks, lengths):
            mask[length:] = False

        self.noise_scheduler.set_timesteps(num_inference_steps)

        for time_step in tqdm(self.noise_scheduler.timesteps, **tqdm_kwargs):
            t = typing.cast(IntTensor, torch.full((batch_size, ), int(time_step), device=self.device))
            predicted_sample: dict[str, Tensor] = self(x_t, t, whole_texts, part_texts, frame_mask=masks)

            # x_t -> x_t-1
            x_t = {                                                                                                                                                        #
                part_name: cast(Tensor,
                                typing.cast(DDIMSchedulerOutput, self.noise_scheduler.step(predicted_sample[part_name], int(time_step), x_t[part_name], eta)).prev_sample)
                for part_name in x_t
            }

        return [{part_name: part_motions[index, :length] for part_name, part_motions in x_t.items()} for index, length in enumerate(lengths)]

    def training_step(self, batch: "BatchData", batch_index: int):
        x_0 = typing.cast(dict[str, FloatTensor], batch.motions)
        part_texts = batch.body_part_texts
        whole_texts = batch.whole_body_texts
        mask = batch.frame_mask
        batch_size = mask.shape[0]

        noise = {part_name: typing.cast(FloatTensor, torch.randn_like(x_0[part_name])) for part_name in self.body_part_names}

        t = typing.cast(IntTensor, torch.randint(0, self.noise_scheduler.config["num_train_timesteps"], (batch_size, ), device=self.device))

        x_t = {part_name: self.noise_scheduler.add_noise(x_0[part_name], noise[part_name], t) for part_name in self.body_part_names}
        predicted_sample: dict[str, Tensor] = self(x_t, t, whole_texts, part_texts, mask)

        part_loss = {f"training/loss/{part_name}": F.mse_loss(x_0[part_name][mask], predicted_sample[part_name][mask]) for part_name in self.body_part_names}
        total_loss = sum(part_loss.values())

        self.log_dict(part_loss)
        self.log("training/loss", total_loss)
        return total_loss

    def validation_step(self, batch: "BatchData", batch_index: int):
        x_0 = typing.cast(dict[str, FloatTensor], batch.motions)
        part_texts = batch.body_part_texts
        whole_texts = batch.whole_body_texts
        mask = batch.frame_mask
        batch_size = mask.shape[0]

        noise = {part_name: typing.cast(FloatTensor, torch.randn_like(x_0[part_name])) for part_name in self.body_part_names}

        t = typing.cast(IntTensor, torch.randint(0, self.noise_scheduler.config["num_train_timesteps"], (batch_size, ), device=self.device))

        x_t = {part_name: self.noise_scheduler.add_noise(x_0[part_name], noise[part_name], t) for part_name in self.body_part_names}
        output: dict[str, Tensor] = self(x_t, t, whole_texts, part_texts, mask)

        part_loss = {f"validation/loss/{part_name}": F.mse_loss(x_0[part_name][mask], output[part_name][mask]) for part_name in self.body_part_names}
        total_loss = sum(part_loss.values())

        self.log_dict(part_loss, batch_size=batch_size, sync_dist=True)
        self.log("validation/loss", total_loss, batch_size=batch_size, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=int(self.trainer.estimated_stepping_batches),
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "name": "learning_rate",
        }

        return [optimizer], [lr_scheduler_config]


class PartMotionEncoder(nn.Module):
    def __init__(
        self,
        part_text_encoder: nn.Module,
        max_frames: int,
        input_dim: int,
        latent_dim: int,
        dropout: float,
        num_heads: int,
        num_conformer_blocks: int,
    ) -> None:
        super().__init__()

        self.part_text_encoder = part_text_encoder

        self.input_projection = nn.Linear(input_dim, latent_dim)

        self.conformer = ConformerEncoder(
            ConformerBlock(
                input_length=max_frames,
                input_dim=latent_dim,
                depth_kernel_size=3,
                feed_forward_expansion=2,
                num_heads=num_heads,
                dropout=dropout,
            ),
            num_conformer_blocks,
        )

    def forward(self, x: Tensor, part_text: list[str], frame_mask: Tensor, time_embedding: Tensor) -> Tensor:
        part_text_embeddings: Tensor = self.part_text_encoder(part_text) #(B, Z)
        x = self.input_projection(x)                                     #(B, F, Z)

        temp = x + (part_text_embeddings + time_embedding)[:, None, :] #(B, F, Z)

        return self.conformer(src=temp, src_key_padding_mask=~frame_mask)


class WholeBodyAttention(nn.Module):
    def __init__(
        self,
        global_text_encoder: nn.Module,
        max_frames: int,
        latent_dim: int,
        body_part_feature_dims: dict[str, int],
        num_heads: int,
        num_layers: int,
        activation_fn: ActivationFn,
        dropout: float,
    ):
        super().__init__()

        self.body_part_names = list(body_part_feature_dims.keys())

        self.global_text_encoder = global_text_encoder

        self.frame_attentions = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=len(self.body_part_names) * latent_dim,
                nhead=num_heads,
                dim_feedforward=1024,
                activation=get_activation_fn(activation_fn),
                dropout=dropout,
                batch_first=True,
            ),
            num_layers,
        )

        self.smooth_net = SmoothNet(max_frames, max_frames, dropout=dropout)

        self.body_part_output_projections = nn.ModuleDict({                     #
            part_name: nn.Linear(latent_dim, body_part_feature_dims[part_name])
            for part_name in self.body_part_names
        })

    def forward(self, x: dict[str, Tensor], whole_body_text: list[str], frame_mask: Tensor) -> dict[str, Tensor]:
        output: Tensor

        whole_text_embeddings: Tensor = self.global_text_encoder(whole_body_text) # (B, Z)

        temp_1 = torch.stack([x[part_name] for part_name in x], dim=-2)             # (B, F, len(parts), Z)
        temp_2 = (temp_1 + whole_text_embeddings[:, None, None, :]).flatten(-2, -1) # (B, F, len(parts)*Z)

        output = temp_2
        output = self.frame_attentions(src=output, src_key_padding_mask=~frame_mask) # (B, F, len(parts)*Z)
        output = output.unflatten(-1, temp_1.shape[-2:]) + temp_1                    # (B, F, len(parts), Z)

        output = self.smooth_net(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # (B, F, len(parts), Z)

        result: dict[str, Tensor] = {                                                           #
            part_name: self.body_part_output_projections[part_name](output[..., part_index, :]) # (B, F, joint_dim)
            for part_index, part_name in enumerate(self.body_part_names)
        }

        return result


from torch.utils.data import DataLoader

from lgtm.dataset.HumanML3D import BodyPart_HumanML3D, BodyPart_HumanML3D_Data, HumanML3D
from lgtm.utils.data_processing import pad_random_truncate_sequences_np


class MotionDiffusion_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        humanml_dataset_dir: Path,
        glove_dir: Path,
        body_part_annotations_path: Path,
        batch_size=32,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.dataset_dir = humanml_dataset_dir
        self.glove_dir = glove_dir
        self.body_part_annotations_path = body_part_annotations_path
        self.batch_size = batch_size

        self.body_part_feature_dims = BodyPart_HumanML3D.body_part_feature_dims
        self.max_frames = 196

    def collate_fn(self, batch: list[BodyPart_HumanML3D_Data]) -> "BatchData":
        part_motions = dict[str, Tensor]()

        frame_mask = None
        for part_name in self.body_part_feature_dims:
            motions = [d.motion[part_name] for d in batch]
            padded_motions, frame_mask = pad_random_truncate_sequences_np(motions, length=self.max_frames)
            part_motions[part_name] = torch.from_numpy(padded_motions)

        body_part_text = {                                #
            part_name: [d.text[part_name] for d in batch]
            for part_name in self.body_part_feature_dims
        }

        frame_mask = torch.from_numpy(frame_mask)

        pos_one_hots = torch.stack([torch.from_numpy(d.original_data.t2m_evaluation_info.pos_one_hots) for d in batch])
        word_embeddings = torch.stack([torch.from_numpy(d.original_data.t2m_evaluation_info.word_embeddings) for d in batch])
        text_lengths = torch.tensor([d.original_data.t2m_evaluation_info.text_lengths for d in batch])

        return BatchData(
            seg_ids=[d.original_data.annotation.seg_id for d in batch],
            motions=part_motions,
            body_part_texts=body_part_text,
            frame_mask=frame_mask,
            whole_body_texts=[d.original_data.annotation.caption for d in batch],
            t2m_evaluation_info=Batch_T2M_EvaluationInfo(
                pos_one_hots,
                word_embeddings,
                text_lengths,
            ),
        )

    def setup(self, stage: Literal["fit", "validate", "test"]):
        if stage == "fit":
            self.train_data = BodyPart_HumanML3D(HumanML3D(self.dataset_dir, self.glove_dir, "train"), self.body_part_annotations_path)
            self.val_data = BodyPart_HumanML3D(HumanML3D(self.dataset_dir, self.glove_dir, "val"), self.body_part_annotations_path)
        elif stage == "validate":
            self.val_data = BodyPart_HumanML3D(HumanML3D(self.dataset_dir, self.glove_dir, "val"), self.body_part_annotations_path)
        elif stage == "test":
            self.test_data = BodyPart_HumanML3D(HumanML3D(self.dataset_dir, self.glove_dir, "test"), self.body_part_annotations_path)
        else:
            raise ValueError("Unsupported stage.")

    def train_dataloader(self):
        dataloader = DataLoader(self.train_data, num_workers=4, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        return cast(DataLoader[BatchData], dataloader)

    def val_dataloader(self):
        dataloader = DataLoader(self.val_data, num_workers=4, batch_size=self.batch_size, collate_fn=self.collate_fn)
        return cast(DataLoader[BatchData], dataloader)

    def test_dataloader(self):
        dataloader = DataLoader(self.test_data, num_workers=1, batch_size=self.batch_size, collate_fn=self.collate_fn)
        return cast(DataLoader[BatchData], dataloader)


@dataclass
class Batch_T2M_EvaluationInfo:
    pos_one_hots: Tensor
    word_embeddings: Tensor
    text_lengths: Tensor


@dataclass
class BatchData:
    seg_ids: list[str]
    motions: dict[str, Tensor]
    body_part_texts: dict[str, list[str]]
    frame_mask: Tensor
    whole_body_texts: list[str]

    t2m_evaluation_info: Batch_T2M_EvaluationInfo


if __name__ == "__main__":
    from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

    class MotionDiffusion_LightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
            parser.link_arguments("data.body_part_feature_dims", "model.body_part_feature_dims", apply_on="instantiate")
            parser.link_arguments("data.max_frames", "model.max_frames", apply_on="instantiate")

    cli = MotionDiffusion_LightningCLI(MotionDiffusion, MotionDiffusion_DataModule)
