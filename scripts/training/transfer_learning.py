#!/usr/bin/env python3
"""
Transfer Learning Training Script
Uses the existing sssalud model as a starting point and fine-tunes on new data
"""

from model.model import captcha_model, model_resnet
from data.datamodule import captcha_dm
import pytorch_lightning as pl
import torch
import os
from utils.config_util import configGetter
from utils.arg_parsers import train_arg_parser

def create_transfer_learning_trainer():
    """Create a trainer for transfer learning"""
    
    # Load configuration
    cfg = configGetter('SOLVER')
    lr = cfg['LR']
    batch_size = cfg['BATCH_SIZE']
    epoch = cfg['EPOCH']
    
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # Create model architecture
    base_model = model_resnet()
    
    # Create the lightning model
    model = captcha_model(model=base_model, lr=lr)
    
    # Load pretrained weights from our sssalud model
    print("🔄 Loading pretrained sssalud model for transfer learning...")
    try:
        checkpoint = torch.load('./checkpoints_sssalud/model.pth', map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print("✅ Successfully loaded pretrained model!")
    except Exception as e:
        print(f"⚠️ Could not load pretrained model: {e}")
        print("Training from scratch instead...")
    
    # Create data module
    dm = captcha_dm(batch_size=batch_size)
    
    return model, dm

def main():
    """Main transfer learning training function"""
    
    print("🎯 Starting Transfer Learning Training...")
    print("=" * 50)
    
    # Create model and data module
    model, dm = create_transfer_learning_trainer()
    
    # Create trainer with optimized settings
    trainer = pl.Trainer(
        deterministic=True,
        devices=1,  # Use 1 GPU
        accelerator='gpu',
        precision='32-true',
        fast_dev_run=False,
        max_epochs=12,  # More epochs for fine-tuning
        log_every_n_steps=50,
        enable_progress_bar=True,
        gradient_clip_val=1.0,  # Add gradient clipping
    )
    
    # Start training
    print("🚀 Beginning transfer learning training...")
    trainer.fit(model, datamodule=dm)
    
    # Save the improved model
    save_dir = './checkpoints_advanced'
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_dir, 'model.pth'))
    
    print("✅ Transfer learning training completed!")
    print(f"📁 Model saved to: {save_dir}/model.pth")

if __name__ == "__main__":
    main()
