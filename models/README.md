# Modelos Progresivos - PyCAPTCHA

Esta carpeta contiene todos los modelos entrenados en orden cronológico y de complejidad.

## Estructura de Modelos

### 01_baseline
- **Descripción**: Modelo inicial básico entrenado con dataset sintético
- **Características**: Modelo simple sin optimizaciones especiales
- **Dataset**: Dataset sintético generado
- **Rendimiento**: Baseline de referencia

### 02_advanced  
- **Descripción**: Modelo mejorado con augmentaciones avanzadas
- **Características**: Augmentaciones de datos, mejores transformaciones
- **Dataset**: Dataset sintético mejorado
- **Rendimiento**: Mejor que baseline

### 03_specialized_sssalud
- **Descripción**: Modelo especializado entrenado con CAPTCHAs reales de sssalud
- **Características**: Transfer learning + fine-tuning con datos reales
- **Dataset**: 10 CAPTCHAs reales de sssalud + sintéticos generados
- **Rendimiento**: 68.3% accuracy, 2 predicciones perfectas de 10
- **Estado**: Mejor modelo individual actual

### 04_ensemble
- **Descripción**: Combinación de múltiples modelos para mejor rendimiento
- **Características**: Votación ponderada entre modelos
- **Componentes**: Specialized (70%) + Advanced (30%)
- **Estado**: En desarrollo - necesita debugging

## Próximos Pasos

1. **Debugging Ensemble**: Corregir predicciones anómalas (zzeeeo)
2. **Post-processing**: Optimizar correcciones automáticas
3. **Más Datos**: Expandir training set con más CAPTCHAs reales
4. **Modelo v4**: Entrenar desde cero con dataset expandido
