from torch import nn
from typing import Literal

class ABMIL(nn.Module):
    def __init__(self, feature_dim, n_heads, head_dim, dropout, gated, hidden_dim, class_num):
        super().__init__()
        self.slide_encoder = ABMILSlideEncoder(
            input_feature_dim = feature_dim,
            n_heads = n_heads,
            head_dim = head_dim,
            dropout = 0.3,
            gated = True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 if class_num==2 else class_num)
        )

    def forward(self, x, which: Literal['logits', 'slide_emb'] = 'logits', device='cuda:1'):
        slide_emb = self.slide_encoder(x,device=device)
        logits = self.classifier(slide_emb)
        if which == 'logits':
            return  logits
        else:
            return slide_emb

#placeholder