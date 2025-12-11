import os
import pandas as pd
import cv2
import torch
from PIL import Image 
from torch.utils.data import Dataset
from src.core import DATASETS, TRANSFORMS
import logging
import numpy as np

@DATASETS.register("ProductDataset")
class ProductDataset(Dataset):
    def __init__(self, data_root, csv_file, img_dir, img_pipeline=None, text_pipeline=None, mode='train'):
        """
        Args:
            data_root (str): 数据集根目录
            csv_file (str): CSV 文件名 (e.g., train.csv)
            img_dir (str): 图片文件夹名
            img_pipeline (list): 图片预处理配置列表
            text_pipeline (list): 文本预处理配置列表
            mode (str): 'train' 或 'test'
        """
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, img_dir)
        csv_path = os.path.join(data_root, csv_file)
        
        if not os.path.exists(csv_path):
             raise FileNotFoundError(f"CSV not found: {csv_path}")
             
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        
        # 构建图片处理管道
        self.img_trans = []
        if img_pipeline:
            self.img_trans = TRANSFORMS.build(img_pipeline)
            
        # 构建文本处理管道
        self.text_trans = None
        if text_pipeline:
            self.text_trans = TRANSFORMS.build(text_pipeline)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. 处理图片 ID 和路径
        img_id = str(row['id'])
        if not img_id.endswith('.jpg'):
            img_id += '.jpg'
            
        img_path = os.path.join(self.img_dir, img_id)
        
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            if image is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        image = Image.fromarray(image)
        
        for t in self.img_trans:
            image = t(image)
            
        text_raw = str(row['title']) + " " + str(row['description'])
        input_ids = torch.zeros(128, dtype=torch.long)
        mask = torch.zeros(128, dtype=torch.long)
        
        if self.text_trans:
            enc = self.text_trans[0](text_raw)
            input_ids = enc['input_ids'].squeeze(0)
            mask = enc['attention_mask'].squeeze(0)
            
        data = {
            'img': image,
            'input_ids': input_ids,
            'attention_mask': mask,
            'id': row['id']
        }
        
        if self.mode != 'test' and 'categories' in row:
            data['label'] = torch.tensor(row['categories'], dtype=torch.long)
            
        return data