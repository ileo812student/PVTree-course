from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class ResNet18Embedding(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.backbone = resnet18(weights=None)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding = nn.Linear(feature_dim, embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embedding(self.backbone(inputs)), dim=1)


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(cosine.dtype)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
