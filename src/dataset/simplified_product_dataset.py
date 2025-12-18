import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from transformers import CLIPTokenizer

# 仅保留必要的组件注册导入（框架必需）
from src.core import DATASETS, TRANSFORMS

# --------------------------
# 全局配置（仅保留CLIP必需的极简配置）
# --------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_MAX_LEN = 77

# --------------------------
# CLIP Tokenizer单例（仅保留必要的，支持多进程）
# --------------------------
def get_clip_tokenizer(model_name=CLIP_MODEL_NAME):
    if not hasattr(get_clip_tokenizer, 'tokenizer'):
        get_clip_tokenizer.tokenizer = CLIPTokenizer.from_pretrained(model_name)
    return get_clip_tokenizer.tokenizer

# --------------------------
# 核心数据集类（仅保留必要的读取和处理逻辑）
# --------------------------
@DATASETS.register("SimplifiedProductDataset")
class SimplifiedProductDataset(Dataset):
    def __init__(self, data_root, csv_file, img_dir, img_pipeline=None, text_pipeline=None, mode='train'):
        # 基础路径配置
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, img_dir)
        self.csv_path = os.path.join(data_root, csv_file)
        self.mode = mode

        # 检查CSV文件存在性
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV文件不存在：{self.csv_path}")

        # 读取CSV（仅保留预计算后的列）
        self.df = pd.read_csv(self.csv_path)

        # 检查预计算列（核心：确保使用预计算的文本）
        if 'text_simplified' not in self.df.columns:
            raise RuntimeError("❌ CSV文件中无text_simplified列！请先执行预计算。")

        # 数据变换（框架必需）
        self.img_trans = TRANSFORMS.build(img_pipeline) if img_pipeline else []
        self.text_trans = TRANSFORMS.build(text_pipeline) if text_pipeline else None

        # CLIP配置（仅保留必要的）
        self.clip_max_len = CLIP_MAX_LEN
        self.tokenizer = get_clip_tokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --------------------------
        # 图片处理（核心极简版）
        # --------------------------
        # 构建图片路径
        img_id = str(row['id'])
        if not img_id.endswith('.jpg'):
            img_id += '.jpg'
        img_path = os.path.join(self.img_dir, img_id)

        # 读取图片（PIL极简版，处理异常）
        try:
            image = Image.open(img_path).convert('RGB') if os.path.exists(img_path) else Image.new('RGB', (224, 224), (0, 0, 0))
        except (IOError, OSError):
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 应用图片变换
        for t in self.img_trans:
            image = t(image)

        # --------------------------
        # 文本处理（仅使用预计算结果，极简版）
        # --------------------------
        text_raw = row['text_simplified']  # 直接读取预计算的文本

        # 初始化CLIP输入张量
        input_ids = torch.zeros(self.clip_max_len, dtype=torch.long)
        attention_mask = torch.zeros(self.clip_max_len, dtype=torch.long)

        # 应用文本变换（如CLIP Tokenize）
        if self.text_trans:
            enc = self.text_trans[0](text_raw)
            input_ids_raw = enc['input_ids'].squeeze(0)
            attention_mask_raw = enc['attention_mask'].squeeze(0)

            # 截断/填充到指定长度
            len_raw = len(input_ids_raw)
            if len_raw > self.clip_max_len:
                input_ids = input_ids_raw[:self.clip_max_len]
                attention_mask = attention_mask_raw[:self.clip_max_len]
            else:
                input_ids[:len_raw] = input_ids_raw
                attention_mask[:len_raw] = attention_mask_raw

        # --------------------------
        # 构建返回数据（核心必需）
        # --------------------------
        data = {
            'img': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'id': row['id']
        }

        # 训练模式添加标签
        if self.mode != 'test' and 'categories' in row:
            data['label'] = torch.tensor(row['categories'], dtype=torch.long)

        return data