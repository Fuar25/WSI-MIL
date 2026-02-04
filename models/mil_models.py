"""Modular MIL model implementations and registry hooks."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.interfaces import Aggregator, BaseMIL, Classifier, DataDict
from core.registry import register_model


class FeatureEncoder(nn.Module):
    """Linear projection + activation encoder for instance features."""

    def __init__(self, input_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, N, C)
        return self.dropout(self.act(self.proj(features)))


class GatedAttention(Aggregator):
    """Standard gated-attention MIL aggregator."""

    def __init__(self, embed_dim: int, attn_dim: Optional[int], gated: bool, dropout: float) -> None:
        super().__init__()
        attn_dim = attn_dim or embed_dim // 2 or 1
        self.gated = gated
        self.attention_a = nn.Linear(embed_dim, attn_dim)
        self.attention_b = nn.Linear(embed_dim, attn_dim) if gated else None
        self.attention_c = nn.Linear(attn_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, instances: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # instances: (B, N, D)
        attn_a = torch.tanh(self.attention_a(instances))
        if self.gated and self.attention_b is not None:
            attn_b = torch.sigmoid(self.attention_b(instances))
            attn_a = attn_a * attn_b
        logits = self.attention_c(self.dropout(attn_a)).squeeze(-1)  # (B, N)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float('-inf'))
        attention = torch.softmax(logits, dim=1)
        bag_embeddings = torch.bmm(attention.unsqueeze(1), instances).squeeze(1)
        return bag_embeddings, attention


class LinearClassifier(Classifier):
    """Single-layer classifier with optional dropout."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, bag_embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(bag_embeddings))


@register_model('mil')
class MILModel(BaseMIL):
    """Configurable MIL wrapper with encoder/aggregator/classifier stages."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        attention_dim: Optional[int] = None,
        gated: bool = True,
        encoder_dropout: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
        external_impl: Optional[nn.Module] = None,
    ) -> None:
        
        super().__init__()
        self.external_impl = external_impl
        enc_dropout = encoder_dropout if encoder_dropout is not None else dropout
        clf_dropout = classifier_dropout if classifier_dropout is not None else dropout

        self.encoder = FeatureEncoder(input_dim, hidden_dim, enc_dropout)
        self.aggregator = GatedAttention(hidden_dim, attention_dim, gated, dropout)
        self.classifier = LinearClassifier(hidden_dim, num_classes, clf_dropout)

    def attach_external_impl(self, module: nn.Module) -> None:
        """Allow plugging in paper-specific implementations later."""
        self.external_impl = module

    def forward(self, data_dict: DataDict) -> Dict[str, torch.Tensor]:
        if self.external_impl is not None:
            return self.external_impl(data_dict)

        features = data_dict['features']  # (B, N, C)
        mask = data_dict.get('mask')  # Optional (B, N) boolean mask
        encoded = self.encoder(features)
        bag_embeddings, attention = self.aggregator(encoded, mask)
        logits = self.classifier(bag_embeddings)

        return {
            'logits': logits,
            'bag_embeddings': bag_embeddings,
            'attention': attention,
        }

