# Modelo Specialized SSsalud (v3) ⭐ MEJOR ACTUAL

## Descripción
Modelo especializado usando transfer learning con CAPTCHAs reales de SSsalud.

## Características
- Arquitectura: ResNet-based con transfer learning
- Dataset: 10 CAPTCHAs reales + sintéticos generados para el dominio específico
- Training: Fine-tuning desde modelo advanced
- Especialización: Adaptado específicamente para CAPTCHAs de SSsalud
- Objetivo: Máxima accuracy en CAPTCHAs reales

## Archivos
- `model.pth`: Checkpoint del modelo especializado

## Uso
```python
from model.model import captcha_model, model_resnet
model = captcha_model(model_resnet())
checkpoint = torch.load('models/03_specialized_sssalud/model.pth')
model.load_state_dict(checkpoint)
```

## Rendimiento ⭐
- **Accuracy**: 68.3% (2 perfectas de 10)
- **CAPTCHAs Perfectas**: 2/10
- **Estado**: Mejor modelo individual actual
- **Evaluado en**: 10 CAPTCHAs reales de SSsalud

## Dataset de Entrenamiento
- 10 CAPTCHAs reales etiquetadas manualmente
- Valores: UKhGh9, cGfFE2, DUzp49, CzSLcN, tZyFmr, 3N4kv3, wvTfzE, z56VDI, 9mLpn5, Z3TUnp

## Análisis de Patrones
- 49 caracteres únicos observados
- Distribución: 40% uppercase, 40% lowercase, 20% dígitos
- Patrones de posición específicos documentados
