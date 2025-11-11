from torch import nn
from typing import Literal

class ABMIL(nn.Module):
    def __init__(self, feature_dim, atte_emb_dim, hidden_dim, class_num):
        super().__init__()
        self.slide_encoder = ABMIL_block(
            input_feature_dim = feature_dim,
            atte_emb_dim= atte_emb_dim,
            hidden_dim = hidden_dim
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

class ABMIL_block(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, atte_emb_dim):
        super().__init__()
        self.input_dim = input_feature_dim
        self.hidden_dim = hidden_dim
        self.atte_emb_dim = atte_emb_dim

        self.pre_emb_layer = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.GELU()
        )

        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, atte_emb_dim),
            nn.Tanh(),
            nn.Linear(atte_emb_dim, 1)
        )

    def forward(self, bags):
        pre_embs = self.pre_emb_layer(bags)
        attention_scores = self.attention_layer(pre_embs)
        weighted_embs = attention_scores.T @ pre_embs

        return weighted_embs
