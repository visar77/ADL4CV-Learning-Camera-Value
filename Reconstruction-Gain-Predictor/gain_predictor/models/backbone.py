"""The context-encoding and candidate-decoding interface for frozen backbones."""
from abc import ABC, abstractmethod

from torch import nn


class FeatureBackbone(nn.Module, ABC):
    """Replaceable frozen extractor. Cameras use OpenCV c2w convention.

    Context state is opaque to callers. Candidate poses are [B,Q,16] and
    normalized intrinsics are [B,Q,4]; decoded features are [B,Q,D].
    """
    feature_dim: int

    @abstractmethod
    def encode_context(self, images, context_poses, mask, context_intrinsics):
        raise NotImplementedError

    @abstractmethod
    def decode_candidates(self, context, candidate_poses, candidate_intrinsics):
        raise NotImplementedError
