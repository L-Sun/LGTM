import torch
from torch import nn
from torch import Tensor
from lgtm.utils.cwd import temporary_change_cwd


class MotionDiffuse_Generator(nn.Module):
    def __init__(self, model_path: str, device) -> None:
        super().__init__()
        # FIXME
        with temporary_change_cwd("/home/haowen/workspace/lpmd/third_packages/MotionDiffuse/text2motion"):
            from third_packages.MotionDiffuse.text2motion.tools.my_generate import Real_MotionDiffuse_Generator
            self.real_gen = Real_MotionDiffuse_Generator(model_path, device)

    @torch.no_grad()
    def generate(self, texts: list[str], lengths: list[int]) -> list[Tensor]:
        return self.real_gen.generate(texts, lengths)


def main():
    generator = MotionDiffuse_Generator("checkpoints/t2m/t2m_motiondiffuse/opt.txt", torch.device("cuda:0"))
    motion = generator.generate(["a man sits down and waving his left hand", "a man flying high in the sky"], [196, 196])

    # FIXME
    with temporary_change_cwd("/home/haowen/workspace/lpmd/third_packages/MotionDiffuse/text2motion"):
        from third_packages.MotionDiffuse.text2motion.tools.my_generate import plot_t2m
        plot_t2m(motion[0].detach().cpu().numpy(), "my_test_waving.gif", "my_test_waving.npy", "test waving", generator.real_gen.opt)
        print("[INFO] Rendered results are saved to /home/haowen/workspace/lpmd/third_packages/MotionDiffuse/text2motion/my_test_waving.gif")
        plot_t2m(motion[1].detach().cpu().numpy(), "my_test_flying.gif", "my_test_flying.npy", "test waving", generator.real_gen.opt)
        print("[INFO] Rendered results are saved to /home/haowen/workspace/lpmd/third_packages/MotionDiffuse/text2motion/my_test_flying.gif")


if __name__ == "__main__":
    main()
