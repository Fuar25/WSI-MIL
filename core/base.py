"""Backward-compatible imports for legacy modules relying on core.base."""

from .interfaces import BaseDataset, BaseModel, BaseMIL, Aggregator, Classifier

__all__ = [
    "BaseDataset",
    "BaseModel",
    "BaseMIL",
    "Aggregator",
    "Classifier",
]
