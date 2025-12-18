import os
import pickle
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from transformers import BlipProcessor
from src.core import DATASETS, TRANSFORMS

@DATASETS.register("BlipProductDataset")
class BlipProductDataset(Dataset):
    def __init__(self, data_root, csv_file, img_dir, processor, mode="train"):
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, img_dir)
        self.df = pd.read_csv(os.path.join(data_root, csv_file))
        self.mode = mode
        self.max_text_len = processor["max_text_len"]
        self.use_augment = processor["use_augment"]
        self.processor = BlipProcessor.from_pretrained(processor["model_name"])
        self.label2id = {label: idx for idx, label in enumerate(sorted(self.df["categories"].unique()))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        # ======================================
        # 关键修改1：缓存逻辑保留，但不一次性加载到内存
        # ======================================
        self.cache_dir = os.path.join(data_root, f"cache_{mode}")
        os.makedirs(self.cache_dir, exist_ok=True)
        # 检查是否有缓存，没有则生成（保留缓存生成逻辑）
        if not os.listdir(self.cache_dir):
            self._cache_data()
        # 只保存缓存文件的路径，不加载数据（核心：延迟加载）
        self.cache_files = [os.path.join(self.cache_dir, f"{idx}.pkl") for idx in range(len(self.df))]

    def _cache_data(self):
        """预处理数据并缓存到磁盘（保留原逻辑，无修改）"""
        for idx in tqdm(range(len(self.df)), desc=f"Caching {self.mode} data"):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, f"{row['id']}.jpg")
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                image = Image.new("RGB", (224, 224), color="white")
            text = f"Title: {row['title']} Description: {row['description']}"
            encoding = self.processor(
                images=image,
                text=text,
                truncation=True,
                max_length=self.max_text_len,
                padding="max_length",
                return_tensors="pt",
            )
            for k, v in encoding.items():
                encoding[k] = v.squeeze(0)
            encoding["label"] = self.label2id[row["categories"]]
            encoding["id"] = row["id"]
            encoding["img"] = encoding["pixel_values"]
            # 保存缓存到磁盘
            pickle.dump(encoding, open(os.path.join(self.cache_dir, f"{idx}.pkl"), "wb"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # ======================================
        # 关键修改2：按需加载缓存文件（用多少加载多少）
        # ======================================
        return pickle.load(open(self.cache_files[idx], "rb"))