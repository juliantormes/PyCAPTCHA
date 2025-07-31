#!/usr/bin/env python3
import torch

print("🔍 VERIFICACIÓN DE GPU")
print("=" * 30)
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Versión PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"Dispositivos GPU: {torch.cuda.device_count()}")
    print(f"GPU actual: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Memoria GPU libre: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
    print("✅ GPU lista para entrenamiento acelerado!")
else:
    print("❌ CUDA no disponible - usando CPU")
