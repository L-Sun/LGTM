import clip
from clip.model import CLIP
from torch import Tensor, nn


def available_models():
    return clip.available_models()


def load_and_freeze_clip(clip_model_name: str, device="cpu") -> CLIP:
    clip_model: CLIP = clip.load(clip_model_name, device=device, jit=False)[0]
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    return clip_model


class CLIP_TextEncoder(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32"):
        super().__init__()
        self.model = load_and_freeze_clip(clip_model_name)
        self.output_dim = self.model.transformer.width

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def encode(self, texts: list[str]):
        device = next(self.parameters()).device
        tokens = clip.tokenize(texts, truncate=True).to(device)
        x: Tensor = self.model.encode_text(tokens)
        return x

    def forward(self, texts: list[str]):
        return self.encode(texts)
