import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
# 保留AMP的新API（适配PyTorch 2.6.0）
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os
import numpy as np
# 新增：抑制dynamo的错误（防止torch.compile残留报错）
import torch._dynamo

torch._dynamo.config.suppress_errors = True  # 关键：禁用dynamo的错误提示
# 新增：禁用transformers的slow image processor提示（彻底消除警告）
import logging

logging.getLogger("transformers.image_processing_utils").setLevel(logging.ERROR)

from src.core import RUNNERS, DATASETS
from src.models.builder import build_model
from src.utils import setup_logger, save_checkpoint


@RUNNERS.register("BlipKFoldRunnerSpeedUp")
class BlipKFoldRunnerSpeedUp:
    def __init__(self, work_dir, epochs, batch_size, lr, n_splits=5, device='cuda'):
        self.work_dir = work_dir
        self.epochs = epochs
        self.batch_size = batch_size
        # ======================================
        # 修改点0：补充lr类型转换（解决之前的类型错误，防止连锁问题）
        # ======================================
        self.lr = float(lr) if lr is not None else 5e-5  # 兜底默认值
        self.n_splits = n_splits
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.logger, self.writer = setup_logger(work_dir)
        # AMP GradScaler（新API，指定cuda）
        self.scaler = GradScaler('cuda') if self.device.type == 'cuda' else None
        self.amp_device = 'cuda' if self.device.type == 'cuda' else 'cpu'

    def train_one_epoch(self, model, loader, optim, criterion, epoch, fold):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"[Fold {fold}] Ep {epoch} Train")
        # ======================================
        # 修改点1：处理loader为空的情况，直接返回0.0
        # ======================================
        if len(loader) == 0:
            self.logger.warning(f"[Fold {fold}] Ep {epoch} Train loader is empty!")
            return 0.0
        try:
            for batch in pbar:
                # non_blocking=True：异步传输，加快CPU→GPU速度
                img = batch.get('img', batch.get('pixel_values')).to(self.device, non_blocking=True)
                ids = batch['input_ids'].to(self.device, non_blocking=True)
                mask = batch['attention_mask'].to(self.device, non_blocking=True)
                label = batch['label'].to(self.device, non_blocking=True)

                optim.zero_grad()

                # 混合精度训练（新API）
                if self.scaler is not None:
                    with autocast(self.amp_device):
                        output = model(img, ids, mask)
                        preds = output['logits'] if isinstance(output, dict) else output
                        loss = criterion(preds, label)
                    # 梯度缩放，避免FP16下溢
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optim)
                    self.scaler.update()
                else:
                    output = model(img, ids, mask)
                    preds = output['logits'] if isinstance(output, dict) else output
                    loss = criterion(preds, label)
                    loss.backward()
                    optim.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            # ======================================
            # 修改点2：确保除法运算安全（再次检查len(loader)）
            # ======================================
            avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
            return avg_loss
        except Exception as e:
            # ======================================
            # 修改点3：捕获训练中的异常，记录日志并返回0.0（避免返回None）
            # ======================================
            self.logger.error(f"[Fold {fold}] Ep {epoch} Train error: {str(e)}", exc_info=True)
            return 0.0

    def validate(self, model, loader, criterion, epoch, fold):
        model.eval()
        correct = 0
        total = 0
        # ======================================
        # 修改点4：处理loader为空的情况，直接返回0.0
        # ======================================
        if len(loader) == 0:
            self.logger.warning(f"[Fold {fold}] Ep {epoch} Val loader is empty!")
            return 0.0
        try:
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"[Fold {fold}] Ep {epoch} Val"):
                    img = batch.get('img', batch.get('pixel_values')).to(self.device, non_blocking=True)
                    ids = batch['input_ids'].to(self.device, non_blocking=True)
                    mask = batch['attention_mask'].to(self.device, non_blocking=True)
                    label = batch['label'].to(self.device, non_blocking=True)

                    # 混合精度验证
                    if self.scaler is not None:
                        with autocast(self.amp_device):
                            output = model(img, ids, mask)
                            preds = output['logits'] if isinstance(output, dict) else output
                    else:
                        output = model(img, ids, mask)
                        preds = output['logits'] if isinstance(output, dict) else output
                    _, predicted = torch.max(preds.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
            # ======================================
            # 修改点5：处理total为0的情况，避免除以零，返回0.0
            # ======================================
            accuracy = correct / total if total > 0 else 0.0
            return accuracy
        except Exception as e:
            # ======================================
            # 修改点6：捕获验证中的异常，记录日志并返回0.0（避免返回None）
            # ======================================
            self.logger.error(f"[Fold {fold}] Ep {epoch} Val error: {str(e)}", exc_info=True)
            return 0.0

    def run(self, cfg, resume_path=None):
        self.logger.info(f"Starting {self.n_splits}-Fold Cross Validation...")

        full_ds = DATASETS.build(cfg['dataset']['train'])
        # 优化：直接取df列，避免逐行遍历
        labels = full_ds.df['categories'].values

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # DataLoader核心优化参数（适配Windows，避免进程问题）
        dataloader_kwargs = {
            'batch_size': self.batch_size,
            # 关键：Windows下num_workers不宜过大，设为2/4即可（避免进程崩溃）
            'num_workers': min(os.cpu_count(), 4),  # 从4改为2，Windows更稳定
            'pin_memory': True,  # 启用内存锁页
            'persistent_workers': True,  # 保持worker进程存活（核心提速）
            'prefetch_factor': 4,  # Windows下prefetch_factor设为2，避免内存溢出
        }

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            fold_id = fold + 1
            self.logger.info(f"========== Fold {fold_id} / {self.n_splits} ==========")

            train_sub = Subset(full_ds, train_idx)
            val_sub = Subset(full_ds, val_idx)

            # 训练集DataLoader
            train_loader = DataLoader(
                train_sub,
                shuffle=True,
                drop_last=True,
                **dataloader_kwargs
            )
            # 验证集DataLoader
            val_loader = DataLoader(
                val_sub,
                shuffle=False,
                drop_last=False,
                **dataloader_kwargs
            )

            # 构建模型（移除torch.compile，因为Windows不支持Triton）
            model = build_model(cfg['model'])
            model.to(self.device)
            # 日志提示：Windows下禁用torch.compile，保留加速逻辑
            self.logger.info("Model loaded (torch.compile disabled for Windows compatibility)")

            # 学习率配置（保持原逻辑）
            backbone_lr = self.lr * 0.1
            head_lr = self.lr

            # 兼容模型参数命名（img_enc/text_enc vs img_backbone/text_backbone）
            # 兼容图像编码器参数
            if hasattr(model, 'img_enc'):
                img_encoder_params = model.img_enc.parameters()
            else:
                img_encoder_params = model.img_backbone.parameters()
            # 兼容文本编码器参数
            if hasattr(model, 'text_enc'):
                text_encoder_params = model.text_enc.parameters()
            else:
                text_encoder_params = model.text_backbone.parameters()

            params = [
                {'params': img_encoder_params, 'lr': backbone_lr},
                {'params': text_encoder_params, 'lr': backbone_lr},
                {'params': model.head.parameters(), 'lr': head_lr}
            ]

            optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
            self.logger.info(f"Optimizer setup: Backbone LR={backbone_lr}, Head LR={head_lr}")

            criterion = nn.CrossEntropyLoss()
            best_acc = 0.0

            for epoch in range(1, self.epochs + 1):
                # 训练和验证
                t_loss = self.train_one_epoch(model, train_loader, optimizer, criterion, epoch, fold_id)
                acc = self.validate(model, val_loader, criterion, epoch, fold_id)

                # ======================================
                # 修改点7：最终兜底，确保t_loss和acc是数值类型（防止极端情况）
                # ======================================
                t_loss = float(t_loss) if t_loss is not None else 0.0
                acc = float(acc) if acc is not None else 0.0

                # 日志输出（现在变量都是数值，可安全格式化）
                self.logger.info(f"Fold {fold_id} Ep {epoch} - Train Loss: {t_loss:.4f}, Val Acc: {acc:.4f}")
                self.writer.add_scalar(f'Fold{fold_id}/Acc', acc, epoch)

                if acc > best_acc:
                    best_acc = acc
                    save_path = os.path.join(self.work_dir, f"fold{fold_id}_best.pth")
                    # 保存模型（无需处理_compile的_orig_mod）
                    torch.save({'state_dict': model.state_dict(), 'acc': best_acc}, save_path)
                    self.logger.info(f"Saved Best Fold {fold_id} model (Acc: {best_acc:.4f})")