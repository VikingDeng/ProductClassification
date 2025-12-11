import torch.nn as nn
import torchvision.models as models
from src.core import BACKBONES

@BACKBONES.register("ResNet50")
class ResNet50(nn.Module):
    def __init__(self, model_name="resnet50", freeze=False):
        super().__init__()
        self.backbone = models.resnet50(weights='DEFAULT')
        
        self.out_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
                
    def forward(self, x):
        return self.backbone(x)
