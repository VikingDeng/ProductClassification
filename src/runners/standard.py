import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.core import RUNNERS, DATASETS
from src.models.builder import build_model
from src.utils import setup_logger, save_checkpoint, load_checkpoint
import os
import logging

@RUNNERS.register("StandardRunner")
class StandardRunner:
    def __init__(self, work_dir, epochs, batch_size, lr, device='cuda'):
        self.work_dir = work_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger, self.writer = setup_logger(work_dir)
        self.best_acc = 0.0

    def train_one_epoch(self, model, loader, optim, criterion, epoch):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch} Train")
        for step, batch in enumerate(pbar):
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
            if step % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), epoch * len(loader) + step)
                pbar.set_postfix({'loss': loss.item()})
        return total_loss / len(loader)

    def validate(self, model, loader, criterion, epoch):
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Epoch {epoch} Val"):
                img = batch['img'].to(self.device)
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                label = batch['label'].to(self.device)
                
                preds = model(img, ids, mask)
                loss = criterion(preds, label)
                val_loss += loss.item()
                
                _, predicted = torch.max(preds.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        
        acc = correct / total
        self.writer.add_scalar('Val/Acc', acc, epoch)
        self.writer.add_scalar('Val/Loss', val_loss / len(loader), epoch)
        return acc

    def run(self, cfg, resume_path=None):
        self.logger.info(f"Using device: {self.device}")
        
        # Build Data
        train_ds = DATASETS.build(cfg['dataset']['train'])
        val_ds = DATASETS.build(cfg['dataset']['val'])
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        model = build_model(cfg['model'])
        model.to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 1

        if resume_path:
            self.logger.info(f"Resuming training from: {resume_path}")
            
            checkpoint = load_checkpoint(model, resume_path)
            
            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    self.logger.info("Optimizer state loaded.")
                except Exception as e:
                    self.logger.warning(f"Failed to load optimizer state: {e}")

            # 恢复 epoch 和 best_acc
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
                
            self.logger.info(f"Resumed successfully. Starting from Epoch {start_epoch}, Best Acc: {self.best_acc:.4f}")
        
        for epoch in range(start_epoch, self.epochs + 1):
            t_loss = self.train_one_epoch(model, train_loader, optimizer, criterion, epoch)
            self.logger.info(f"Epoch {epoch} Train Loss: {t_loss:.4f}")
            
            acc = self.validate(model, val_loader, criterion, epoch)
            self.logger.info(f"Epoch {epoch} Val Acc: {acc:.4f}")
            
            is_best = acc > self.best_acc
            if is_best: self.best_acc = acc
            
            save_checkpoint(
                {
                    'epoch': epoch, 
                    'state_dict': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'best_acc': self.best_acc
                },
                self.work_dir,
                is_best=is_best
            )
