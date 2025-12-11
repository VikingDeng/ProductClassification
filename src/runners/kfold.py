import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os
import numpy as np
from src.core import RUNNERS, DATASETS
from src.models.builder import build_model
from src.utils import setup_logger, save_checkpoint

@RUNNERS.register("KFoldRunner")
class KFoldRunner:
    def __init__(self, work_dir, epochs, batch_size, lr, n_splits=5, device='cuda'):
        self.work_dir = work_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_splits = n_splits
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger, self.writer = setup_logger(work_dir)

    def train_one_epoch(self, model, loader, optim, criterion, epoch, fold):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"[Fold {fold}] Ep {epoch} Train")
        for batch in pbar:
            img = batch['img'].to(self.device)
            ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            label = batch['label'].to(self.device)
            
            optim.zero_grad()
            preds = model(img, ids, mask)
            loss = criterion(preds, label)
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        return total_loss / len(loader)

    def validate(self, model, loader, criterion, epoch, fold):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"[Fold {fold}] Ep {epoch} Val"):
                img = batch['img'].to(self.device)
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                label = batch['label'].to(self.device)
                
                preds = model(img, ids, mask)
                _, predicted = torch.max(preds.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        return correct / total

    def run(self, cfg, resume_path=None):
        self.logger.info(f"Starting {self.n_splits}-Fold Cross Validation...")

        full_ds = DATASETS.build(cfg['dataset']['train'])
        labels = [full_ds.df.iloc[i]['categories'] for i in range(len(full_ds))]
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            fold_id = fold + 1
            self.logger.info(f"========== Fold {fold_id} / {self.n_splits} ==========")

            train_sub = Subset(full_ds, train_idx)
            val_sub = Subset(full_ds, val_idx)
            
            train_loader = DataLoader(train_sub, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
            val_loader = DataLoader(val_sub, batch_size=self.batch_size, shuffle=False, num_workers=4)
            
            model = build_model(cfg['model'])
            model.to(self.device)
            

            backbone_lr = self.lr * 0.1
            head_lr = self.lr
            
            params = [
                {'params': model.img_enc.parameters(), 'lr': backbone_lr},
                {'params': model.text_enc.parameters(), 'lr': backbone_lr},
                {'params': model.head.parameters(), 'lr': head_lr}
            ]
            
            optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
            self.logger.info(f"Optimizer setup: Backbone LR={backbone_lr}, Head LR={head_lr}")


            criterion = nn.CrossEntropyLoss()
            
            best_acc = 0.0
            
            for epoch in range(1, self.epochs + 1):
                t_loss = self.train_one_epoch(model, train_loader, optimizer, criterion, epoch, fold_id)
                acc = self.validate(model, val_loader, criterion, epoch, fold_id)
                
                self.logger.info(f"Fold {fold_id} Ep {epoch} - Train Loss: {t_loss:.4f}, Val Acc: {acc:.4f}")
                self.writer.add_scalar(f'Fold{fold_id}/Acc', acc, epoch)

                if acc > best_acc:
                    best_acc = acc
                    save_path = os.path.join(self.work_dir, f"fold{fold_id}_best.pth")
                    torch.save({'state_dict': model.state_dict(), 'acc': best_acc}, save_path)
                    self.logger.info(f"Saved Best Fold {fold_id} model (Acc: {best_acc:.4f})")