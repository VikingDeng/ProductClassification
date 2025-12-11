import torch
import torch.nn as nn
from src.core import HEADS

@HEADS.register("ConcatHead")
class ConcatHead(nn.Module):
    def __init__(self, img_dim, text_dim, num_classes, dropout=0.3):
        super().__init__()
        self.bn = nn.BatchNorm1d(img_dim + text_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(img_dim + text_dim, num_classes)

    def forward(self, img_feat, text_feat):
        x = torch.cat([img_feat, text_feat], dim=1)
        x = self.bn(x)
        x = self.drop(x)
        return self.fc(x)
