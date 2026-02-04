from typing import Dict

import torch
import torch.nn as nn

from core.interfaces import BaseModel, DataDict
from core.registry import register_model

@register_model('linear_probe')
class LinearProbe(BaseModel):
    """
    Linear Probe model for WSI classification using pre-computed WSI-level features.
    """
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int = 1, 
                 dropout: float = 0,
                 **kwargs):
        """
        Args:
            input_dim: Dimension of input features.
            num_classes: Number of output classes.
            dropout: Dropout rate.
            **kwargs: Ignored arguments (to maintain compatibility with ABMIL signature).
        """
        super(LinearProbe, self).__init__()
        
        self.linear = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, data_dict: DataDict) -> Dict[str, torch.Tensor]:
        features = data_dict['features']
        if features.dim() == 3 and features.size(1) == 1:
            features = features.squeeze(1)
        features = self.dropout(features)
        logits = self.linear(features)
        return {'logits': logits}
