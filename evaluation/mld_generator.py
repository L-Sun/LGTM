import sys
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch import Tensor


class MLD_Generator(nn.Module):
    def __init__(self, model_path: Path, progress=True) -> None:
        super().__init__()
        
        # --- mld imports --- #
        sys.path.append("/home/haowen/workspace/lpmd")  # FIXME
        sys.path.append("third_packages/mld")
        import torch
        from third_packages.mld.mld.config import parse_args
        from third_packages.mld.mld.data.get_data import get_datasets
        from third_packages.mld.mld.models.get_model import get_model
        from third_packages.mld.mld.utils.logger import create_logger

        # --- mld args --- #
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]

        sys.argv.append("--cfg")
        sys.argv.append("third_packages/mld/configs/config_mld_humanml3d.yaml")
        sys.argv.append("--cfg_assets")
        sys.argv.append("third_packages/mld/configs/assets.yaml")

        # --- mld process --- #
        import os
        import omegaconf
        cfg = parse_args(phase="test", config_path=Path(os.getcwd()) / "third_packages/mld/configs/base.yaml")  # parse config file
        cfg.FOLDER = cfg.TEST.FOLDER

        def check_and_fix(path_dir):
            import os
            from pathlib import Path
            p = Path(path_dir)
            if not p.is_dir():
                print(f"[WARNING] `{p}` is not a valid directory, ", end='')
                if path_dir[:2] == "./":
                    p = Path(path_dir[2:])
                p = Path(os.getcwd()) / f"third_packages/mld" / p
                print(f"changed to `{p}` instead.")
                path_dir = str(p)
            return path_dir
        
        def update_cfg_path(cfg_dict):
            changed_cfg = {}
            try:  # skip interpolation keys
                items = cfg_dict.items()
            except:
                return
            for k, v in items:
                if isinstance(v, str) and v[:2] == "./":
                    changed_cfg[k] = check_and_fix(v)
                elif isinstance(v, dict) or isinstance(v, omegaconf.dictconfig.DictConfig):
                    update_cfg_path(v)  # update path recursively
            cfg_dict.update(changed_cfg)

        update_cfg_path(cfg)

        # create logger
        logger = create_logger(cfg, phase="test")

        # create dataset
        datasets = get_datasets(cfg, logger=logger, phase="test")[0]

        # create model
        model = get_model(cfg, datasets)
        state_dict = torch.load(model_path,
                                map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        model.sample_mean = cfg.TEST.MEAN  # False
        model.fact = cfg.TEST.FACT  # 1

        self.old_feats2joints = model.feats2joints

        def new_feats2joints(feature):
            # [B, F, J*Z]
            B, F, _ = feature.shape
            return [feature[i].view(F, 22, -1) for i in range(B)]
        model.feats2joints = lambda feature: feature  # return feature
        model.eval()

        self.model = model
        self.progress = progress

        # --- recover argv --- #
        sys.argv = original_argv

    @torch.no_grad()
    def generate(self, texts: list[str], lengths: list[int], return_feat=False) -> list[Tensor]:
        # FIXME self.progress not used
        batch = {}
        batch['text'] = texts
        batch['length'] = lengths
        return self.model(batch, return_feat)


def main():
    sys.path.append("/home/haowen/workspace/lpmd")  # FIXME
    from pathlib import Path
    import numpy as np
    from tqdm import tqdm
    from lpmd.dataset.HumanML3D import HumanML3D
    from lpmd.utils.transform import exchange_yz
    from lpmd.utils.visualization import animate

    mld_generator = MLD_Generator(Path("third_packages/mld/checkpoints/mld_humanml3d_checkpoint/1222_mld_humanml3d_FID041.ckpt"))

    texts = [
        "a man walking forward", # 0
        "a man jogging",
        "a man running",
        "a man sitting down and boxing",
        "a man jumping up happily",
        "a man climbing up a ladder", # 5
        "a man flying in the sky",
        "a man kicks something or someone with his left leg", # 7
        "the standing person kicks with their left foot before going back to their original stance.",
        "he is flying kick with his left leg",
        "a person walks, turns slight to the right, squats, puts hand on both knees while squatting, and then squats again",
    ]
    rets = mld_generator.generate(texts, [196] * len(texts))
    dataset = HumanML3D(Path("./data/HumanML3D"), Path("./data/glove"), "test")
    for i, r in enumerate(rets):
        print(f"Processing [{texts[i]}] ... ")
        motion = r.detach().cpu().numpy()
        x, y, z = motion[:, :, 0], motion[:, :, 1], motion[:, :, 2]
        motion = np.stack([x, z, y], axis=2)
        animate(motion, dataset.parents, 20).save(f"test_{i:02d}.mp4")
        print("Done. ")


if __name__ == "__main__":
    main()
