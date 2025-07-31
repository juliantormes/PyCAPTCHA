#!/usr/bin/env python3
"""
Script de entrenamiento rápido independiente para GPU
Optimizado para RTX 3070 con CUDA
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import string
import argparse
from torch.amp import GradScaler, autocast
import time

# Configuración
FULL_CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
NUM_CLASSES = len(FULL_CHARSET)  # 62 clases
CAPTCHA_LENGTH = 6
HEIGHT = 80
WIDTH = 230

class SimpleCaptchaDataset(Dataset):
    """Dataset simple para CAPTCHAs sintéticos"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        
        # Obtener lista de archivos
        self.image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.png')]
        print(f"📁 {split}: {len(self.image_files)} muestras encontradas")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Extraer label del nombre del archivo
        label_str = self.image_files[idx].split('.')[0]  # "0007Xn" de "0007Xn.94812.png"
        
        # Convertir a índices
        label_indices = []
        for char in label_str:
            if char in FULL_CHARSET:
                label_indices.append(FULL_CHARSET.index(char))
            else:
                print(f"⚠️  Carácter desconocido: {char} en {self.image_files[idx]}")
                label_indices.append(0)  # Usar 0 como fallback
        
        # Completar o truncar a 6 caracteres
        while len(label_indices) < CAPTCHA_LENGTH:
            label_indices.append(0)
        label_indices = label_indices[:CAPTCHA_LENGTH]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label_indices, dtype=torch.long)

class FastCaptchaModel(nn.Module):
    """Modelo rápido con ResNet-18 para GPU"""
    
    def __init__(self, num_classes=NUM_CLASSES, captcha_length=CAPTCHA_LENGTH):
        super().__init__()
        
        # Backbone ResNet-18
        import torchvision.models as models
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Modificar la última capa
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes * captcha_length)
        
        self.num_classes = num_classes
        self.captcha_length = captcha_length
        
    def forward(self, x):
        # x: [batch_size, 3, height, width]
        features = self.backbone(x)  # [batch_size, num_classes * captcha_length]
        
        # Reshape para separar posiciones
        batch_size = features.size(0)
        outputs = features.view(batch_size, self.captcha_length, self.num_classes)
        
        return outputs  # [batch_size, captcha_length, num_classes]

def accuracy_fn(outputs, targets):
    """Calcular precisión por carácter y exacta"""
    # outputs: [batch_size, captcha_length, num_classes]
    # targets: [batch_size, captcha_length]
    
    predicted = torch.argmax(outputs, dim=2)  # [batch_size, captcha_length]
    
    # Precisión por carácter
    char_correct = (predicted == targets).float()
    char_accuracy = char_correct.mean().item()
    
    # Precisión exacta (toda la secuencia correcta)
    sequence_correct = char_correct.all(dim=1)  # [batch_size]
    exact_accuracy = sequence_correct.float().mean().item()
    
    return char_accuracy, exact_accuracy

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Entrenar por una época"""
    model.train()
    total_loss = 0
    total_char_acc = 0
    total_exact_acc = 0
    
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass con mixed precision
        with autocast(device_type='cuda'):
            outputs = model(images)  # [batch_size, captcha_length, num_classes]
            
            # Calcular loss para cada posición
            loss = 0
            for i in range(CAPTCHA_LENGTH):
                loss += criterion(outputs[:, i, :], targets[:, i])
            loss /= CAPTCHA_LENGTH
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Métricas
        with torch.no_grad():
            char_acc, exact_acc = accuracy_fn(outputs, targets)
            total_loss += loss.item()
            total_char_acc += char_acc
            total_exact_acc += exact_acc
        
        # Log cada 50 batches
        if batch_idx % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:2d} [{batch_idx:4d}/{len(train_loader):4d}] "
                  f"Loss: {loss.item():.4f} | Char: {char_acc:.3f} | Exact: {exact_acc:.3f} | "
                  f"Time: {elapsed:.1f}s")
    
    return total_loss / len(train_loader), total_char_acc / len(train_loader), total_exact_acc / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validar modelo"""
    model.eval()
    total_loss = 0
    total_char_acc = 0
    total_exact_acc = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            
            # Loss
            loss = 0
            for i in range(CAPTCHA_LENGTH):
                loss += criterion(outputs[:, i, :], targets[:, i])
            loss /= CAPTCHA_LENGTH
            
            # Métricas
            char_acc, exact_acc = accuracy_fn(outputs, targets)
            total_loss += loss.item()
            total_char_acc += char_acc
            total_exact_acc += exact_acc
    
    return total_loss / len(val_loader), total_char_acc / len(val_loader), total_exact_acc / len(val_loader)

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento rápido con GPU')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Directorio del dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='Tamaño del batch')
    parser.add_argument('--epochs', type=int, default=15, help='Número de épocas')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='models/fast_gpu', help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Dispositivo: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Datasets
    print("📁 Cargando datasets...")
    train_dataset = SimpleCaptchaDataset(args.data_dir, 'train', transform)
    val_dataset = SimpleCaptchaDataset(args.data_dir, 'val', transform)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"🎯 Entrenamiento: {len(train_dataset)} muestras, {len(train_loader)} batches")
    print(f"🎯 Validación: {len(val_dataset)} muestras, {len(val_loader)} batches")
    
    # Modelo
    print("🤖 Inicializando modelo...")
    model = FastCaptchaModel(NUM_CLASSES, CAPTCHA_LENGTH).to(device)
    
    # Optimizer y scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr * 10,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    # Loss y scaler
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    print(f"🚀 Iniciando entrenamiento: {args.epochs} épocas, batch_size={args.batch_size}")
    print("=" * 80)
    
    # Entrenamiento
    best_exact_acc = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_char_acc, train_exact_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch + 1
        )
        
        # Validate
        val_loss, val_char_acc, val_exact_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch results
        print(f"\n📊 EPOCH {epoch + 1:2d} RESULTS:")
        print(f"   Train - Loss: {train_loss:.4f} | Char: {train_char_acc:.3f} | Exact: {train_exact_acc:.3f}")
        print(f"   Val   - Loss: {val_loss:.4f} | Char: {val_char_acc:.3f} | Exact: {val_exact_acc:.3f}")
        print(f"   LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_exact_acc > best_exact_acc:
            best_exact_acc = val_exact_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_exact_acc': best_exact_acc,
                'char_acc': val_char_acc,
                'val_loss': val_loss
            }
            
            model_path = os.path.join(args.output_dir, f'best_model_epoch_{epoch+1:02d}_acc_{val_exact_acc:.3f}.pth')
            torch.save(checkpoint, model_path)
            print(f"   💾 Modelo guardado: {model_path}")
        
        print("-" * 80)
    
    print(f"🎉 ENTRENAMIENTO COMPLETADO!")
    print(f"📈 Mejor precisión exacta: {best_exact_acc:.3f}")
    
    # Test rápido
    print(f"\n🧪 PRUEBA RÁPIDA:")
    model.eval()
    with torch.no_grad():
        # Tomar un batch de validación
        images, targets = next(iter(val_loader))
        images, targets = images.to(device), targets.to(device)
        
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=2)
        
        # Mostrar primeras 5 predicciones
        for i in range(min(5, len(predicted))):
            true_text = ''.join([FULL_CHARSET[idx] for idx in targets[i].cpu().numpy()])
            pred_text = ''.join([FULL_CHARSET[idx] for idx in predicted[i].cpu().numpy()])
            match = "✅" if true_text == pred_text else "❌"
            print(f"   {match} Real: {true_text} | Pred: {pred_text}")

if __name__ == "__main__":
    main()
