import os
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.core import RUNNERS, DATASETS
from src.models.builder import build_model
import logging

@RUNNERS.register("TestRunner")
class TestRunner:
    def __init__(self, work_dir, batch_size=32, output_file="submission.csv", **kwargs):
        self.work_dir = work_dir
        self.batch_size = batch_size
        self.output_file = output_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)

    def load_models(self, cfg):
        models = []
        ckpt_files = [
            f for f in os.listdir(self.work_dir) 
            if (f.endswith('_best.pth') or f == 'best_model.pth')
        ]
        
        if not ckpt_files:
            raise FileNotFoundError(f"No .pth files found in {self.work_dir}")

        self.logger.info(f"Found {len(ckpt_files)} models for ensemble: {ckpt_files}")

        for ckpt_file in ckpt_files:
            model = build_model(cfg['model'])
            model.to(self.device)
            model.eval()
            
            ckpt_path = os.path.join(self.work_dir, ckpt_file)
            self.logger.info(f"Loading weights from {ckpt_path}...")
            
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(new_state_dict, strict=False)
            models.append(model)
            
        return models

    def run(self, cfg, **kwargs):
        self.logger.info("Starting Test/Inference Mode...")
        
        if 'test' not in cfg['dataset']:
            raise ValueError("Config must have 'dataset.test' section")
            
        test_ds = DATASETS.build(cfg['dataset']['test'])
        test_loader = DataLoader(
            test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        models = self.load_models(cfg)
        
        results = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                img = batch['img'].to(self.device)
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                img_ids = batch['id']
                
                avg_probs = None
                for model in models:
                    logits = model(img, ids, mask)
                    probs = F.softmax(logits, dim=1)
                    
                    if avg_probs is None:
                        avg_probs = probs
                    else:
                        avg_probs += probs
                
                avg_probs /= len(models)
                preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
                
                for img_id, pred_cls in zip(img_ids, preds):
                    results.append({'id': img_id, 'categories': pred_cls})

        save_path = os.path.join(self.work_dir, self.output_file)
        df = pd.DataFrame(results)
        df['id'] = df['id'].astype(str).str.replace('.jpg', '')
        df.to_csv(save_path, index=False)
        self.logger.info(f"Submission saved to: {save_path}")