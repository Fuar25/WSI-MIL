
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from core.base import BaseModel

class ABMIL(BaseModel):
    """
    Attention-based Multiple Instance Learning (ABMIL) model.
    """
    def __init__(self, 
                 input_dim: int = 768, 
                 hidden_dim: int = 256, 
                 num_classes: int = 1, 
                 n_heads: int = 1, 
                 dropout: float = 0.2, 
                 gated: bool = True):
        super(ABMIL, self).__init__()
        
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        if gated:
            self.gate_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Sigmoid(),
                nn.Dropout(dropout)
            )
        else:
            self.gate_net = None
            
        self.attention_weights = nn.Linear(hidden_dim, n_heads)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * n_heads, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_instances, input_dim)
        """
        # x shape: [B, N, D]
        
        # Attention mechanism
        a = self.attention_net(x)  # [B, N, H]
        
        if self.gate_net is not None:
            g = self.gate_net(x)   # [B, N, H]
            a = a * g
            
        # Compute attention scores
        attn_scores = self.attention_weights(a) # [B, N, K] (K=n_heads)
        attn_scores = torch.transpose(attn_scores, 2, 1)  # [B, K, N]
        attn_scores = F.softmax(attn_scores, dim=2)  # Softmax over instances
        
        # Aggregation
        # [B, K, N] x [B, N, D] -> [B, K, D]
        m = torch.matmul(attn_scores, x)
        
        # Flatten heads if K > 1
        m = m.view(m.size(0), -1)  # [B, K*D]
        
        # Classification
        logits = self.classifier(m)
        
        return logits, attn_scores


# You can add more models here easily.
# Example:
# class MeanPoolingMIL(nn.Module):
#     ...
