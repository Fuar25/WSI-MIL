import torch
import torch.nn as nn
from core.base import BaseModel

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
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) or (batch_size, 1, input_dim)
        """
        # Handle case where input has an extra dimension [B, 1, D]
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
            
        # Apply dropout
        x = self.dropout(x)
        
        # Linear projection
        logits = self.linear(x)
        
        return logits, None  # Return None for attention scores
