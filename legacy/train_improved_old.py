#!/usr/bin/env python3
"""
Improved training script with advanced techniques
"""
import os
import sys
import importlib.util
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Import custom_losses dynamically
custom_losses_file = os.path.join('scripts', 'training', 'custom_losses.py')
spec = importlib.util.spec_from_file_location("custom_losses", custom_losses_file)
custom_losses_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_losses_module)
CombinedLoss = custom_losses_module.CombinedLoss

from model.model import CaptchaModel
from data.datamodule import CaptchaDataModule

class ImprovedCaptchaModel(CaptchaModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = CombinedLoss()
        
        # Learning rate scheduling
        self.learning_rate = 1e-3
        self.warmup_steps = 1000
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self(images)
        
        # Use custom loss
        loss = self.loss_fn(predictions, targets)
        
        # Calculate accuracy per position
        pred_chars = torch.argmax(predictions, dim=-1)
        correct = (pred_chars == targets).float()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', correct.mean(), prog_bar=True)
        
        # Log position-wise accuracy
        for pos in range(6):
            pos_acc = correct[:, pos].mean()
            self.log(f'train_acc_pos_{pos}', pos_acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self(images)
        
        loss = self.loss_fn(predictions, targets)
        pred_chars = torch.argmax(predictions, dim=-1)
        correct = (pred_chars == targets).float()
        
        # Perfect match accuracy (all characters correct)
        perfect_match = (correct.sum(dim=1) == 6).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', correct.mean(), prog_bar=True)
        self.log('val_perfect_match', perfect_match, prog_bar=True)
        
        return loss

def train_improved_model():
    # Set up data
    data_module = CaptchaDataModule(
        data_dir="./dataset_sssalud",
        batch_size=32,
        num_workers=4
    )
    
    # Model
    model = ImprovedCaptchaModel()
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints_improved",
        filename="improved-{epoch:02d}-{val_perfect_match:.3f}",
        monitor="val_perfect_match",
        mode="max",
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val_perfect_match",
        mode="max",
        patience=5,
        verbose=True
    )
    
    # Logger
    logger = TensorBoardLogger("lightning_logs", name="improved_model")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=15,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Effective batch size = 64
        val_check_interval=0.5,  # Validate twice per epoch
    )
    
    # Train
    trainer.fit(model, data_module)
    
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    train_improved_model()
