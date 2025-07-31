# Modelo Baseline (v1)

## Descripción
Primer modelo entrenado como punto de referencia básico.

## Características
- Arquitectura: ResNet-based
- Dataset: Sintético generado
- Augmentaciones: Básicas
- Objetivo: Establecer baseline de rendimiento

## Archivos
- `model.pth`: Checkpoint del modelo entrenado

## Uso
```python
from model.model import captcha_model, model_resnet
model = captcha_model(model_resnet())
checkpoint = torch.load('models/01_baseline/model.pth')
model.load_state_dict(checkpoint)
```

## Rendimiento
- Accuracy en sintéticos: TBD
- Accuracy en CAPTCHAs reales: Bajo (esperado)
