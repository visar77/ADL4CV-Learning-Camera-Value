"""Trainable heads mapping frozen embeddings and optional camera geometry to gain."""
from torch import Tensor, nn
import torch

from gain_predictor.geometry import (
    COVERAGE_DIM,
    RELATIVE_POSE_DIM,
    camera_coverage,
    relative_camera_poses,
)


class FeatureGainHead(nn.Sequential):
    def __init__(self, feature_dim: int, dropout: float = 0.0,
                 hidden_dim: int = 64, hidden_layers: int = 1):
        if hidden_dim < 1 or hidden_layers < 1:
            raise ValueError("Gain-head width and hidden-layer count must be positive.")
        layers = []
        input_dim = feature_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        super().__init__(*layers)


class CoverageGainHead(nn.Module):
    """Linear camera-coverage estimate plus a candidate-conditioned correction."""

    def __init__(self, input_dim: int, dropout: float = 0.0,
                 hidden_dim: int = 32, hidden_layers: int = 1):
        super().__init__()
        self.coverage_gain = nn.Linear(COVERAGE_DIM, 1, bias=False)
        self.correction = FeatureGainHead(input_dim, dropout, hidden_dim, hidden_layers)

    def forward(self, inputs: Tensor) -> Tensor:
        coverage = inputs[..., -COVERAGE_DIM:]
        return self.coverage_gain(coverage) + self.correction(inputs)


class PoseGainHead(nn.Module):
    """Encode each full context-to-candidate pose before pooling across views."""

    def __init__(self, embedding_dim: int, dropout: float = 0.0,
                 hidden_dim: int = 32, hidden_layers: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        relation_width = 16
        self.pose_encoder = nn.Sequential(nn.Linear(RELATIVE_POSE_DIM, relation_width), nn.GELU())
        summary_dim = 2 * relation_width + 1
        self.pose_gain = nn.Linear(summary_dim, 1, bias=False)
        self.correction = FeatureGainHead(embedding_dim + summary_dim, dropout, hidden_dim, hidden_layers)

    def forward(self, inputs: Tensor) -> Tensor:
        embedding = inputs[:, :self.embedding_dim]
        views = inputs[:, self.embedding_dim:].reshape(len(inputs), -1, RELATIVE_POSE_DIM + 1)
        valid = views[..., -1:].bool()
        tokens = self.pose_encoder(views[..., :-1])
        count = valid.sum(1).to(tokens.dtype)
        mean = tokens.masked_fill(~valid, 0).sum(1) / count
        maximum = tokens.masked_fill(~valid, -torch.inf).amax(1)
        summary = torch.cat((mean, maximum, count.log()), dim=-1)
        return self.pose_gain(summary) + self.correction(torch.cat((embedding, summary), dim=-1))


FEATURE_HEADS = {"mlp": FeatureGainHead, "coverage": CoverageGainHead, "pose": PoseGainHead}
DEFAULT_HEAD_WIDTH = {"mlp": 64, "coverage": 32, "pose": 32}


def feature_head_inputs(head_type, embeddings, poses, candidate, mask):
    if head_type == "pose":
        relations = relative_camera_poses(poses, candidate, mask)
        return torch.cat((embeddings, relations.flatten(1)), dim=-1)
    if head_type == "coverage":
        coverage = camera_coverage(poses, candidate, mask)
        return torch.cat((embeddings, coverage), dim=-1)
    return embeddings
