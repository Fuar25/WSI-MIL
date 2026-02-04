"""Core interface definitions for datasets, MIL models, and modular components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn as nn

DataDict = Dict[str, Tensor]


class BaseDataset(Dataset, ABC):
    """All datasets must return a standardized dictionary for each index."""

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a dict with keys such as 'features', 'coords', 'label', etc."""

    @abstractmethod
    def __len__(self) -> int:
        pass


class BaseModel(nn.Module, ABC):
    """Base model interface enforcing dict-based forward signatures."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, data_dict: DataDict) -> Dict[str, Tensor]:
        """
        Args:
            data_dict: {'features': Tensor, 'coords': Tensor|None, ...}
                       features: (B, N, C) where N is instance count and C the channel size.
        Returns:
            dict with at least {'logits': Tensor of shape (B, num_classes)}.
        """


class BaseMIL(BaseModel, ABC):
    """Specialized base class for MIL models with encoder/aggregator/classifier stages."""

    @abstractmethod
    def forward(self, data_dict: DataDict) -> Dict[str, Tensor]:
        """Must call component stack and expose attention if available."""


class Aggregator(nn.Module, ABC):
    """Aggregates instance-level embeddings into bag-level descriptors."""

    @abstractmethod
    def forward(
        self, instances: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            instances: (B, N, D) bag of embeddings.
            mask: Optional (B, N) mask indicating valid instances.
        Returns:
            A tuple of (bag_embeddings, attention_weights|None).
        """


class Classifier(nn.Module, ABC):
    """Maps bag-level embeddings to logits."""

    @abstractmethod
    def forward(self, bag_embeddings: Tensor) -> Tensor:
        """
        Args:
            bag_embeddings: (B, D) aggregated features.
        Returns:
            logits: (B, num_classes)
        """
*** End of File
