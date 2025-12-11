import torch.nn as nn
from src.core import BACKBONES, HEADS

class MultiModalNet(nn.Module):
    def __init__(self, img_cfg, text_cfg, head_cfg):
        super().__init__()
        
        self.img_enc = BACKBONES.build(img_cfg)
        self.text_enc = BACKBONES.build(text_cfg)
        
        dynamic_args = {
            'img_dim': self.img_enc.out_dim,
            'text_dim': self.text_enc.out_dim
        }
        
        self.head = HEADS.build(head_cfg, **dynamic_args)

    def forward(self, img, input_ids, attention_mask):
        i_f = self.img_enc(img)
        t_f = self.text_enc(input_ids, attention_mask)
        return self.head(i_f, t_f)

def build_model(cfg):
    return MultiModalNet(cfg['img_backbone'], cfg['text_backbone'], cfg['head'])