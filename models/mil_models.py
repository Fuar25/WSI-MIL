
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.builder import create_model
from config.config import runtime_config

class MILModel(nn.Module):
    """
    Generic Wrapper for MIL-Lab models.
    Supports both custom initialization and loading pretrained weights.
    """
    def __init__(self, 
                 model_name: str = 'abmil', 
                 num_classes: int = 1, 
                 pretrained: bool = False,
                 checkpoint_path: str = '',
                 input_dim: int = 1024, 
                 hidden_dim: int = 256, 
                 dropout: float = 0.2, 
                 gated: bool = True,
                 n_heads: int = 1,
                 **kwargs):
        super(MILModel, self).__init__()
        
        self.model_name = model_name
        
        # Heuristic: if model_name contains dots (e.g. 'abmil.base.uni.pc108-24k'), 
        # it's likely a specific MIL-Lab model string that might have pretrained weights.
        is_predefined_string = '.' in model_name
        
        if pretrained and is_predefined_string:
            # Pretrained mode: Load from Hugging Face or local checkpoint using the specific model string
            # We avoid passing structural params (input_dim, hidden_dim) to prevent conflicts with pretrained config
            print(f"Loading pretrained MIL-Lab model: {model_name}")
            self.model = create_model(
                model_name=model_name,
                num_classes=num_classes,
                checkpoint_path=checkpoint_path,
                from_pretrained=True,
                **kwargs
            )
        else:
            # Custom mode: Initialize from scratch using provided parameters
            # We map WSI-ABMIL parameters to MIL-Lab parameters
            print(f"Initializing custom MIL-Lab model: {model_name}")
            
            # Common parameters mapping
            mil_kwargs = {
                'in_dim': input_dim,
                'embed_dim': hidden_dim,
                'dropout': dropout,
                'gate': gated,
                'num_classes': num_classes,
                'checkpoint_path': checkpoint_path,
                'from_pretrained': False,
            }
            
            # Add any extra kwargs (e.g. k_sample for CLAM)
            mil_kwargs.update(kwargs)
            
            self.model = create_model(
                model_name=model_name,
                **mil_kwargs
            )
        
    def forward(self, x, label=None, loss_fn=None):
        """
        Args:
            x: (batch_size, num_instances, input_dim)
            label: Optional labels for internal loss calculation (e.g. CLAM instance loss)
            loss_fn: Optional loss function for internal loss calculation
        """
        # MIL-Lab forward expects: h, return_attention=True
        # Some models (like CLAM) can compute loss internally if label and loss_fn are provided
        results_dict, log_dict = self.model(x, label=label, loss_fn=loss_fn, return_attention=True)
        
        logits = results_dict['logits']
        loss = results_dict.get('loss', None)
        
        # Extract attention scores
        # MIL-Lab models usually return raw attention scores (before softmax) in log_dict['attention']
        attn_scores = log_dict.get('attention', None)
        
        if attn_scores is not None:
            # Apply softmax to match WSI-ABMIL expectation (if it expects probabilities)
            # Note: Some MIL-Lab models might already return softmaxed scores, but ABMIL returns raw.
            # We assume raw scores for consistency.
            if isinstance(attn_scores, torch.Tensor):
                if attn_scores.dim() >= 2:
                    attn_scores = F.softmax(attn_scores, dim=-1)
            
        return logits, loss, attn_scores

