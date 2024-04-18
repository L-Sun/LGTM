import random

import torch
from torch import nn
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.image.fid import _compute_fid


class Diversity(Metric):
    def __init__(self, diversity_times: int = 200):
        super().__init__()

        self.diversity_times = diversity_times

        self.features: list[Tensor]
        self.add_state("features", default=[], dist_reduce_fx="cat")

    def update(self, features: Tensor):
        self.features.append(features)

    def compute(self) -> dict[str, float]:
        features = torch.cat(self.features, dim=0)

        num_samples = features.shape[0]
        first_indices = torch.multinomial(torch.ones(num_samples), self.diversity_times, replacement=False)
        second_indices = torch.multinomial(torch.ones(num_samples), self.diversity_times, replacement=False)

        dist: Tensor = torch.linalg.norm(features[first_indices] - features[second_indices], dim=1)
        return {"diversity": dist.mean().item()}


class FrechetInceptionDistance(Metric):
    def __init__(self, feature_dims: int) -> None:
        super().__init__()

        self.real_features_sum: Tensor
        self.real_features_cov_sum: Tensor
        self.real_features_num_samples: Tensor
        self.add_state("real_features_sum", torch.zeros(feature_dims).double(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", torch.zeros(feature_dims, feature_dims).double(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.fake_features_sum: Tensor
        self.fake_features_cov_sum: Tensor
        self.fake_features_num_samples: Tensor
        self.add_state("fake_features_sum", torch.zeros(feature_dims).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", torch.zeros(feature_dims, feature_dims).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

    def update(self, features: Tensor, real: bool):

        if features.dim() == 1:
            features = features.unsqueeze(0)

        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.T @ features
            self.real_features_num_samples += features.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.T @ features
            self.fake_features_num_samples += features.shape[0]

    def compute(self) -> dict[str, float]:
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")

        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.T @ mean_real
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.T @ mean_fake
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)

        return {"fid": _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).item()}

    def reset_fake(self):
        self.fake_features_sum.zero_()
        self.fake_features_cov_sum.zero_()
        self.fake_features_num_samples.zero_()

    def reset_real(self):
        self.real_features_sum.zero_()
        self.real_features_cov_sum.zero_()
        self.real_features_num_samples.zero_()


class RetrievalPrecision(Metric):
    def __init__(self, top_k: int, bin_size: int = 32):
        super().__init__()

        self.top_k = top_k
        self.bin_size = bin_size

        self.queries: list[Tensor]
        self.values: list[Tensor]
        self.add_state("queries", default=[], dist_reduce_fx="cat")
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, queries: Tensor, values: Tensor):
        self.queries.append(queries)
        self.values.append(values)

    def compute(self) -> dict[str, float]:
        queries = torch.cat(self.queries, dim=0)
        values = torch.cat(self.values, dim=0)

        pairs = [(query, value) for query, value in zip(queries, values)]
        random.shuffle(pairs)

        queries = torch.stack([query for query, _ in pairs], dim=0)
        values = torch.stack([value for _, value in pairs], dim=0)

        device = queries.device

        num_samples = queries.shape[0]
        num_bins = num_samples // self.bin_size

        queries = queries[:num_bins * self.bin_size]
        values = values[:num_bins * self.bin_size]

        queries = queries.reshape(num_bins, self.bin_size, -1)
        values = values.reshape(num_bins, self.bin_size, -1)

        dist_mat = torch.cdist(queries, values, p=2) # (num_bins, bin_size, bin_size)

        correct_indices = torch.arange(self.bin_size, device=device)[None].expand(num_bins, -1)[..., None] # (num_bins, bin_size, 1)
        top_k_indices = dist_mat.topk(self.top_k, dim=2, largest=False).indices                            # (num_bins, bin_size, top_k)

        correct = [torch.any(top_k_indices[..., :k + 1] == correct_indices, dim=2).sum() for k in range(self.top_k)]

        R_precision = [correct[k] / (num_bins * self.bin_size) for k in range(self.top_k)]

        return {
            **{
                f"R_precision_{k+1}": R_precision[k].item()
                for k in range(self.top_k)
            },
            "Matching_Score": torch.einsum("...ii", dist_mat).sum().item() / (num_bins * self.bin_size),
        }


class Similarity(Metric):
    def __init__(self):
        super().__init__()

        self.similarities: list[Tensor]
        self.add_state("similarities", default=[], dist_reduce_fx="cat")

    def update(self, x: Tensor, y: Tensor):
        x_logits = nn.functional.normalize(x, dim=-1)
        y_logits = nn.functional.normalize(y, dim=-1)

        similarities = torch.sum(x_logits * y_logits, dim=-1) / 2.0 + 0.5
        self.similarities.append(similarities)

    def compute(self) -> dict[str, float]:
        similarities = torch.cat(self.similarities, dim=0)

        return {"similarity": similarities.mean(dim=0).item()}
