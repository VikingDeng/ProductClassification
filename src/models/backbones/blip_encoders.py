import torch.nn as nn
from transformers import BlipVisionModel, BlipTextModel  # 导入BLIP的视觉/文本模型
from src.core import BACKBONES
import logging


@BACKBONES.register("BlipVision")
class BlipVision(nn.Module):
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", freeze=True):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading BLIP Vision: {model_name}")

        # 加载BLIP视觉模型（默认使用BLIP-base，适配商品分类场景）
        self.model = BlipVisionModel.from_pretrained(model_name)

        # 定义输出维度：BLIP视觉模型的隐藏层维度（固定为768，对应BLIP-base）
        self.out_dim = self.model.config.hidden_size

        # 冻结参数逻辑（和CLIP保持一致）
        if freeze:
            self.logger.info("Freezing BLIP Vision parameters")
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        前向传播：接收图像特征（pixel_values），返回CLS池化特征
        Args:
            x (torch.Tensor): 输入的图像张量，形状为 [batch_size, 3, H, W]（即pixel_values）
        Returns:
            torch.Tensor: BLIP视觉模型的池化特征，形状为 [batch_size, 768]
        """
        # BLIP视觉模型的输出包含pooler_output（池化后的CLS特征），和CLIP的pooler_output逻辑一致
        outputs = self.model(x)
        # 若部分BLIP版本无pooler_output，可替换为：outputs.last_hidden_state[:, 0, :]（取CLS token的特征）
        return outputs.pooler_output


@BACKBONES.register("BlipText")
class BlipText(nn.Module):
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", freeze=True):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading BLIP Text: {model_name}")

        # 加载BLIP文本模型（和视觉模型使用同一基础模型，保证特征兼容性）
        self.model = BlipTextModel.from_pretrained(model_name)
        # 定义输出维度：BLIP文本模型的隐藏层维度（固定为768，对应BLIP-base）
        self.out_dim = self.model.config.hidden_size

        # 冻结参数逻辑（和CLIP保持一致）
        if freeze:
            self.logger.info("Freezing BLIP Text parameters")
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        前向传播：接收文本的input_ids和attention_mask，返回CLS特征
        Args:
            input_ids (torch.Tensor): 文本的token id张量，形状为 [batch_size, seq_len]
            attention_mask (torch.Tensor): 文本的注意力掩码，形状为 [batch_size, seq_len]
        Returns:
            torch.Tensor: BLIP文本模型的CLS特征，形状为 [batch_size, 768]
        """
        # BLIP文本模型的输出为last_hidden_state，取第一个token（CLS）作为特征
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # 对应CLIPText的text_embeds，返回维度为[batch_size, 768]的特征
        return outputs.last_hidden_state[:, 0, :]