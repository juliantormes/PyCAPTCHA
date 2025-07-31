#!/usr/bin/env python3
"""
Entrenamiento rápido con dataset optimizado para RTX 3070
Diseñado para pruebas rápidas y validación del pipeline
"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import string
from train_enhanced_model import SyntheticDataset, EnhancedCaptchaModel

# Configuración optimizada para entrenamiento rápido
FULL_CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
CLASS_NUM = len(FULL_CHARSET)  # 62 clases
CHAR_LEN = 6
HEIGHT = 80   # Usar dimensiones reales de CAPTCHAs
WIDTH = 230

def create_fast_data_loaders(batch_size=256):  # Batch grande para GPU
    """Data loaders optimizados para entrenamiento rápido"""
    
    # Transformaciones más simples para velocidad
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Datasets
    train_dataset = SyntheticDataset('dataset_optimized/train', transform=transform)
    val_dataset = SyntheticDataset('dataset_optimized/val', transform=transform)
    
    # Data loaders optimizados para RTX 3070
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=6,          # Más workers para RTX 3070
        persistent_workers=True,
        pin_memory=True,        # Optimización GPU
        drop_last=True,
        prefetch_factor=4       # Prefetch para overlapping
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader

class FastCaptchaModel(EnhancedCaptchaModel):
    """Modelo optimizado para entrenamiento rápido"""
    
    def __init__(self, lr=2e-3):  # Learning rate más alto
        super().__init__(lr)
        
        # Usar dimensiones reales
        self.example_input_array = torch.randn(1, 3, HEIGHT, WIDTH)
        
    def configure_optimizers(self):
        # Optimizador más agresivo para entrenamiento rápido
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Scheduler para convergencia rápida
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=1000,  # Ajustar según dataset
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

def train_fast_model():
    """Entrenamiento rápido con dataset optimizado"""
    
    print("🚀 ENTRENAMIENTO RÁPIDO CON GPU RTX 3070")
    print("=" * 50)
    
    # Verificar dataset
    train_dir = 'dataset_optimized/train'
    val_dir = 'dataset_optimized/val'
    
    if not os.path.exists(train_dir):
        print("❌ Dataset optimizado no encontrado. Ejecuta optimized_captcha_generator.py")
        return
    
    # Verificar GPU
    if not torch.cuda.is_available():
        print("⚠️ CUDA no disponible, usando CPU (será más lento)")
    else:
        print(f"✅ Usando GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Data loaders optimizados
    train_loader, val_loader = create_fast_data_loaders(batch_size=256)
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")
    
    # Modelo optimizado
    model = FastCaptchaModel(lr=2e-3)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/05_fast_gpu',
        filename='fast-epoch={epoch}-val_acc={val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=2,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_acc',
        patience=3,
        mode='max',
        verbose=True
    )
    
    # Trainer optimizado para RTX 3070
    trainer = pl.Trainer(
        max_epochs=10,  # Pocas épocas para prueba rápida
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',  # Mixed precision para RTX 3070
        log_every_n_steps=10,
        val_check_interval=50,  # Validar cada 50 batches
        gradient_clip_val=1.0,
        deterministic=False,   # Permitir operaciones rápidas
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    print("🎯 Iniciando entrenamiento rápido...")
    print(f"⚡ Precisión mixta: {'Activada' if torch.cuda.is_available() else 'Desactivada'}")
    print(f"🔥 Batch size: 256")
    print(f"📈 Learning rate: 2e-3")
    
    # Entrenar
    trainer.fit(model, train_loader, val_loader)
    
    print("🎉 Entrenamiento rápido completado!")
    print(f"📁 Modelos guardados en: models/05_fast_gpu/")
    
    # Mostrar estadísticas de GPU
    if torch.cuda.is_available():
        print(f"💾 Memoria GPU usada: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        print(f"💾 Memoria GPU reservada: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")

if __name__ == "__main__":
    train_fast_model()
