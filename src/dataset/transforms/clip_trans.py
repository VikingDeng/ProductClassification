import torchvision.transforms as T
from transformers import CLIPProcessor
from src.core import TRANSFORMS
import logging

@TRANSFORMS.register("CLIPImageProcessor")
class CLIPImageProcessor:
    def __init__(self, model_name="openai/clip-vit-base-patch32", use_augment=False):
        
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.use_augment = use_augment

        self.aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomRotation(degrees=10),
        ])
    
    def __call__(self, img):
        if self.use_augment:
            img = self.aug(img)
            
        out = self.processor(images=img, return_tensors="pt")
        
        return out['pixel_values'][0]

@TRANSFORMS.register("CLIPTextTokenizer")
class CLIPTextTokenizer:
    def __init__(self, model_name="openai/clip-vit-base-patch32", max_len=77):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.max_len = max_len
        
    def __call__(self, text):
        return self.processor(
            text=text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )