# Ensemble Models (v4)

## Descripción
Combinación inteligente de múltiples modelos para maximizar accuracy.

## Características
- **Método**: Votación ponderada entre modelos
- **Componentes**: 
  - Specialized SSsalud (70% peso)
  - Advanced (30% peso)
- **Algoritmo**: Weighted voting por posición de carácter
- **Objetivo**: Superar rendimiento de modelos individuales

## Archivos
- `ensemble_predictor.py`: Predictor principal con múltiples métodos de votación
- `simple_ensemble.py`: Versión simplificada usando subprocess

## Estado Actual ⚠️
- **Problema**: Predicciones anómalas (zzeeeo, zeeeeo)
- **Causa**: Posible incompatibilidad en carga de modelos
- **Solución en desarrollo**: Post-processing + debugging

## Uso
```python
from models.ensemble.ensemble_predictor import EnsemblePredictor

model_paths = [
    'models/03_specialized_sssalud/model.pth',
    'models/02_advanced/model.pth'
]
weights = [0.7, 0.3]

ensemble = EnsemblePredictor(model_paths, weights)
prediction, confidence = ensemble.predict_ensemble('image.png')
```

## Métodos de Votación
- `weighted_voting`: Peso por modelo (recomendado)
- `majority_voting`: Voto mayoritario simple
- `confidence_based`: Basado en confianza del modelo

## Próximos Pasos
1. Debuggear predicciones anómalas
2. Implementar post-processing robusto
3. Evaluar en 20 CAPTCHAs completas
