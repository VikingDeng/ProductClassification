import torch.nn as nn
from transformers import AutoModel
from src.core import BACKBONES
import logging

@BACKBONES.register("BertBase")
class BertBase(nn.Module):
    def __init__(self, model_name="bert-base-uncased", freeze=True):
        super().__init__()
        logger = logging.getLogger(__name__)

        logger.info(f"Loading HF Model: {model_name}")
        self.bert = AutoModel.from_pretrained(model_name)
        self.out_dim = self.bert.config.hidden_size
        
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]
