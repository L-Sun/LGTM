from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch import nn as nn

from lgtm.utils.cwd import temporary_change_cwd


class TMR_Wrapper(nn.Module):
    def __init__(self, model_dir: Path):
        super().__init__()

        import sys
        sys.path.append("third_packages/TMR")

        from hydra.utils import instantiate

        from third_packages.TMR.src.config import read_config
        from third_packages.TMR.src.data.collate import collate_x_dict
        from third_packages.TMR.src.data.text import TokenEmbeddings
        from third_packages.TMR.src.load import load_model_from_cfg
        from third_packages.TMR.src.model.tmr import TMR

        self.collate_x_dict = collate_x_dict

        cfg = read_config(str(model_dir))
        cfg.run_dir = model_dir.relative_to("third_packages/TMR")

        with temporary_change_cwd("third_packages/TMR"):
            self.text_model: TokenEmbeddings = instantiate(cfg.data.text_to_token_emb)
            self.tmr_model: TMR = load_model_from_cfg(cfg)

            if model_dir.parts[-1] == "tmr_humanml3d_guoh3dfeats":
                self.mean = torch.load("stats/humanml3d/guoh3dfeats/mean.pt")
                self.std = torch.load("stats/humanml3d/guoh3dfeats/std.pt")
            else:
                self.mean = torch.load(f"stats/humanml3d_{model_dir.parts[-1]}/guoh3dfeats/mean.pt")
                self.std = torch.load(f"stats/humanml3d_{model_dir.parts[-1]}/guoh3dfeats/std.pt")

        self.latent_dim = 256

    def normalize(self, motion: Tensor) -> Tensor:
        return (motion - self.mean.to(motion.device)) / self.std.to(motion.device)

    def encode_motion(self, motions: list[Tensor]) -> Tensor:

        device = next(self.parameters()).device

        motion_x_dicts = [{"x": motion, "length": len(motion)} for motion in motions]

        motion_x_dict = self.collate_x_dict(motion_x_dicts, device=str(device))

        latent = cast(Tensor, self.tmr_model.encode(motion_x_dict, sample_mean=True))

        return latent

    def encode_text(self, texts: list[str]) -> Tensor:
        device = next(self.parameters()).device
        text_x_dict = self.collate_x_dict(self.text_model(texts), device=str(device))
        latent = cast(Tensor, self.tmr_model.encode(text_x_dict, sample_mean=True))
        return latent

    def forward(self, x: list[Tensor] | list[str]):
        if isinstance(x[0], Tensor):
            return self.encode_motion(cast(list[Tensor], x))
        else:
            return self.encode_text(cast(list[str], x))
