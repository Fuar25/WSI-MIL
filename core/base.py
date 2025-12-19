from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    """
    Abstract base class for datasets.
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for models.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass
