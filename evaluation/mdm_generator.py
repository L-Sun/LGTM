import sys
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch import Tensor
from loguru import logger


class MDM_Generator(nn.Module):
    def __init__(self, model_path: Path, progress=True) -> None:
        super().__init__()

        sys.path.append("third_packages/mdm")

        from third_packages.mdm.model.cfg_sampler import ClassifierFreeSampleModel
        from third_packages.mdm.model.mdm import MDM
        from third_packages.mdm.sample.generate import load_dataset
        from third_packages.mdm.utils.model_util import create_gaussian_diffusion, get_model_args, load_model_wo_clip
        from third_packages.mdm.utils.parser_util import generate_args
        from third_packages.mdm.data_loaders.tensors import collate

        # add model_path to args
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]

        sys.argv.append("--model_path")
        sys.argv.append(str(model_path))

        args = generate_args()
        self.args = args

        self.progress = progress

        max_frames = 196 if args.dataset in ["kit", "humanml"] else 60

        self.dataset = load_dataset(args, max_frames, None)
        self.model = MDM(**get_model_args(args, self.dataset))
        self.diffusion = create_gaussian_diffusion(args)

        logger.info(f"Loading MDM model from {model_path} ...")
        state_dict = torch.load(model_path, map_location="cpu")
        load_model_wo_clip(self.model, state_dict)

        if args.guidance_param != 1:
            self.model = ClassifierFreeSampleModel(self.model)

        self.collate = collate

        # recover argv
        sys.argv = original_argv

    def generate(self, texts: list[str], lengths: list[int]) -> list[Tensor]:
        batch_size = len(texts)

        device = next(self.parameters()).device

        collate_args = [{
            'inp': torch.zeros(length),
            'tokens': None,
            'lengths': length,
            'text': text,
        } for text, length in zip(texts, lengths)]

        _, model_kwargs = self.collate(collate_args)
        max_frames = model_kwargs['y']['lengths'].max().item()

        if self.args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(batch_size, device=device) * self.args.guidance_param

        samples = self.diffusion.p_sample_loop(
            self.model,
            (batch_size, self.model.njoints, self.model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=self.progress,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        samples = cast(Tensor, samples)
        samples = samples.squeeze().permute(0, 2, 1) # (bs, num_frames, num_features)
        results = [sample[:length] for sample, length in zip(samples, lengths)]
        return results


if __name__ == "__main__":
    mdm_generator = MDM_Generator(Path("third_packages/mdm/save/humanml_trans_enc_512/model000200000.pt"))

    mdm_generator.generate(["a man walks forward"] * 2, [196] * 2)
