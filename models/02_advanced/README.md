# Modelo Advanced (v2)

## Descripción  
Versión mejorada con augmentaciones avanzadas y mejor preprocessing.

## Características
- Arquitectura: ResNet-based mejorado
- Dataset: Sintético con mejor generación
- Augmentaciones: Avanzadas (rotación, blur, noise, etc.)
- Regularización: Dropout mejorado
- Objetivo: Mejor generalización

## Archivos
- `model.pth`: Checkpoint del modelo entrenado

## Uso
```python
from model.model import captcha_model, model_resnet
model = captcha_model(model_resnet())
checkpoint = torch.load('models/02_advanced/model.pth')
model.load_state_dict(checkpoint)
```

## Rendimiento
- Accuracy en sintéticos: Mejorado vs baseline
- Accuracy en CAPTCHAs reales: Limitado (solo entrenado con sintéticos)
