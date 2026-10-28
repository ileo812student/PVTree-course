# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PVTree-course contributors.
# Course-project addition; see LICENSE for terms and README for upstream attribution.
import numpy as np
import torch

from recognition.metrics import verification_metrics
from recognition.model import ArcFaceHead, ResNet18Embedding
from recognition.train import state_dict_sha256


def test_arcface_forward_backward_and_embeddings():
    model = ResNet18Embedding(32)
    head = ArcFaceHead(32, 4, scale=64, margin=0.5)
    images = torch.randn(4, 3, 128, 128)
    labels = torch.arange(4)
    embeddings = model(images)
    assert embeddings.shape == (4, 32)
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(4), atol=1e-5)
    loss = torch.nn.functional.cross_entropy(head(embeddings, labels), labels)
    loss.backward()
    assert torch.isfinite(loss)


def test_verification_metrics():
    result = verification_metrics(np.array([0.95, 0.85, 0.2, -0.1]), np.array([1, 1, 0, 0]))
    assert result["eer"] == 0.0
    assert result["tar_at_far_1e-3"] == 1.0
    assert result["tar_at_far_1e-4"] is None


def test_initial_state_hash_is_seed_reproducible_and_value_sensitive():
    torch.manual_seed(2026)
    first = ResNet18Embedding(32)
    first_hash = state_dict_sha256(first)
    torch.manual_seed(2026)
    second = ResNet18Embedding(32)
    assert state_dict_sha256(second) == first_hash
    with torch.no_grad():
        next(second.parameters()).view(-1)[0].add_(1)
    assert state_dict_sha256(second) != first_hash
