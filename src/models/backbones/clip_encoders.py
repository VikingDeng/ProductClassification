import torch.nn as nn
from transformers import CLIPVisionModel, CLIPTextModelWithProjection
from src.core import BACKBONES
import logging

@BACKBONES.register("CLIPVision")
class CLIPVision(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze=True):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading CLIP Vision: {model_name}")

        self.model = CLIPVisionModel.from_pretrained(model_name)

        self.out_dim = self.model.config.hidden_size
        
        if freeze:
            self.logger.info("Freezing CLIP Vision parameters")
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.model(x).pooler_output

@BACKBONES.register("CLIPText")
class CLIPText(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze=True):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading CLIP Text: {model_name}")

        self.model = CLIPTextModelWithProjection.from_pretrained(model_name)
        self.out_dim = self.model.config.projection_dim
        
        if freeze:
            self.logger.info("Freezing CLIP Text parameters")
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).text_embeds