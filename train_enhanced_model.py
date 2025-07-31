#!/usr/bin/env python3
"""
Script de entrenamiento para modelo CAPTCHA con 62 clases
Vocabulario completo: 0-9, a-z, A-Z
"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torchvision.models as models
from data.datamodule import CaptchaDataModule
import string

# Configuración del vocabulario completo
FULL_CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
CLASS_NUM = len(FULL_CHARSET)  # 62 clases
CHAR_LEN = 6  # 6 caracteres por CAPTCHA
HEIGHT = 64
WIDTH = 224

print(f"📚 Vocabulario completo: {FULL_CHARSET}")
print(f"📏 Total clases: {CLASS_NUM}")

class EnhancedCaptchaModel(pl.LightningModule):
    """Modelo mejorado para 62 clases"""
    
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        
        # Backbone ResNet-18
        self.resnet = models.resnet18(weights=None)
        
        # Capa clasificadora para 62 clases x 6 posiciones
        self.resnet.fc = nn.Linear(512, CHAR_LEN * CLASS_NUM)
        
        # Función de pérdida
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"🏗️ Modelo creado: ResNet-18 con {CHAR_LEN * CLASS_NUM} outputs")
    
    def forward(self, x):
        x = self.resnet(x)
        # Reshape para 6 posiciones x 62 clases
        x = x.view(x.size(0), CHAR_LEN, CLASS_NUM)
        return x
    
    def str_to_indices(self, text: str):
        """Convertir texto a índices según el vocabulario completo"""
        indices = []
        for char in text:
            if char in FULL_CHARSET:
                indices.append(FULL_CHARSET.index(char))
            else:
                # Carácter no válido, usar 0 por defecto
                indices.append(0)
        return indices
    
    def indices_to_str(self, indices):
        """Convertir índices a texto"""
        text = ""
        for idx in indices:
            if 0 <= idx < len(FULL_CHARSET):
                text += FULL_CHARSET[idx]
            else:
                text += "?"
        return text
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # Forward pass
        outputs = self(images)  # [batch, 6, 62]
        
        # Calcular pérdida para cada posición
        total_loss = 0
        for pos in range(CHAR_LEN):
            pos_output = outputs[:, pos, :]  # [batch, 62]
            pos_target = labels[:, pos]      # [batch]
            total_loss += self.criterion(pos_output, pos_target)
        
        # Calcular precisión
        predicted = torch.argmax(outputs, dim=2)  # [batch, 6]
        correct = (predicted == labels).all(dim=1).float().mean()
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_acc', correct, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        outputs = self(images)
        
        # Calcular pérdida
        total_loss = 0
        for pos in range(CHAR_LEN):
            pos_output = outputs[:, pos, :]
            pos_target = labels[:, pos]
            total_loss += self.criterion(pos_output, pos_target)
        
        # Calcular precisión
        predicted = torch.argmax(outputs, dim=2)
        correct = (predicted == labels).all(dim=1).float().mean()
        
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_acc', correct, prog_bar=True)
        
        # Log de ejemplos en el primer batch
        if batch_idx == 0 and len(predicted) > 0:
            for i in range(min(3, len(predicted))):
                pred_text = self.indices_to_str(predicted[i].cpu().numpy())
                true_text = self.indices_to_str(labels[i].cpu().numpy())
                print(f"  Pred: '{pred_text}' | True: '{true_text}'")
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

class SyntheticDataset(torch.utils.data.Dataset):
    """Dataset para CAPTCHAs sintéticos con 62 clases"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Cargar lista de archivos
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        print(f"📁 Dataset cargado: {len(self.image_files)} imágenes desde {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        # Extraer etiqueta del nombre del archivo
        label_text = self.image_files[idx].split('_')[0]  # "ABC123_000001.png" -> "ABC123"
        
        # Convertir a índices
        label_indices = []
        for char in label_text:
            if char in FULL_CHARSET:
                label_indices.append(FULL_CHARSET.index(char))
            else:
                label_indices.append(0)  # Carácter por defecto
        
        # Asegurar longitud correcta
        while len(label_indices) < CHAR_LEN:
            label_indices.append(0)
        label_indices = label_indices[:CHAR_LEN]
        
        return image, torch.tensor(label_indices, dtype=torch.long)

def create_data_loaders(batch_size=32):
    """Crear data loaders para entrenamiento y validación"""
    
    import torchvision.transforms as transforms
    
    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Datasets
    train_dataset = SyntheticDataset('dataset_synthetic_v2/train', transform=transform)
    val_dataset = SyntheticDataset('dataset_synthetic_v2/val', transform=transform)
    
    # Data loaders optimizados para GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,  # Para transferencia GPU más rápida
        drop_last=True    # Para batches consistentes
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_model():
    """Entrenar el modelo mejorado"""
    
    print("🚀 ENTRENAMIENTO DE MODELO CON 62 CLASES")
    print("=" * 50)
    
    # Verificar dataset
    train_dir = 'dataset_synthetic_v2/train'
    val_dir = 'dataset_synthetic_v2/val'
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("❌ Dataset no encontrado. Ejecuta primero advanced_captcha_generator.py")
        return
    
    # Crear data loaders optimizados
    train_loader, val_loader = create_data_loaders(batch_size=128)  # Batch más grande para GPU
    print(f"✅ Data loaders creados")
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")
    
    # Crear modelo
    model = EnhancedCaptchaModel(lr=1e-3)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/04_enhanced_62classes',
        filename='epoch={epoch}-val_acc={val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True
    )
    
    # Trainer optimizado para GPU
    trainer = pl.Trainer(
        max_epochs=15,  # Reducido para entrenamiento más rápido
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='gpu',  # Usar GPU
        devices=1,
        precision='16-mixed',  # Mixed precision para velocidad
        log_every_n_steps=25,
        val_check_interval=0.25,  # Validar cada cuarto de época
        gradient_clip_val=1.0,    # Gradient clipping para estabilidad
        deterministic=False,      # Permitir operaciones no deterministas para velocidad
    )
    
    print("🎯 Iniciando entrenamiento...")
    
    # Entrenar
    trainer.fit(model, train_loader, val_loader)
    
    print("🎉 Entrenamiento completado!")
    print(f"📁 Modelos guardados en: models/04_enhanced_62classes/")

if __name__ == "__main__":
    train_model()
