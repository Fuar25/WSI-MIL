import torch
from torch import nn
import torch.nn.functional as F

class MeanMIL(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, class_num):
        super().__init__()
        self.slide_encoder = MeanMIL_block(input_feature_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1 if class_num == 2 else class_num)
        )

    def forward(self, features):
        slide_emb = self.slide_encoder(features)
        logits = self.classifier(slide_emb)
        return logits

class ABMIL(nn.Module):
    def __init__(self, feature_dim, atte_emb_dim, hidden_dim, class_num):
        super().__init__()
        self.slide_encoder = ABMIL_block(
            input_feature_dim = feature_dim,
            atte_emb_dim= atte_emb_dim,
            hidden_dim = hidden_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1 if class_num==2 else class_num)
        )

    def forward(self, features):
        slide_emb = self.slide_encoder(features)
        logits = self.classifier(slide_emb)
        return logits


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
        bags = bags.squeeze(0)
        pre_embs = self.pre_emb_layer(bags)
        attention_scores = F.softmax(self.attention_layer(pre_embs), dim=0)
        weighted_embs = attention_scores.T @ pre_embs

        return weighted_embs

class MeanMIL_block(nn.Module):
    def __init__(self, input_feature_dim, output_dim):
        super().__init__()
        self.pre_emb_layer = nn.Linear(input_feature_dim, output_dim)

    def forward(self, bags):
        bags = bags.squeeze(0)
        pre_embs = self.pre_emb_layer(bags)
        embs = torch.mean(pre_embs, dim=0, keepdim=True)

        return embs