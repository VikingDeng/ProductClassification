import torch
import torch.nn as nn
from src.core import HEADS

@HEADS.register("CrossAttentionHead")
class CrossAttentionHead(nn.Module):
    def __init__(self, img_dim, text_dim, num_classes, hidden_dim=512, num_heads=8, dropout=0.3):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, img_feat, text_feat):
        img_emb = self.img_proj(img_feat).unsqueeze(1)
        text_emb = self.text_proj(text_feat).unsqueeze(1)

        attn_out, _ = self.attn(query=text_emb, key=img_emb, value=img_emb)

        fusion = torch.cat([text_emb, attn_out], dim=-1).squeeze(1)
        
        # 分类
        return self.fc(fusion)