# Experimentos y Análisis

Esta carpeta contiene todos los scripts de análisis y experimentación.

## Estructura

### analysis/
Scripts para analizar patrones en CAPTCHAs reales:
- `analyze_extended_patterns.py`: Análisis completo de 20 CAPTCHAs
- `analyze_sssalud_patterns.py`: Análisis inicial de 10 CAPTCHAs

### postprocessing/
Experimentos con post-procesamiento:
- `captcha_postprocessor.py`: Sistema de corrección automática

## Resultados Clave

### Análisis de 20 CAPTCHAs
- **Caracteres únicos**: 49 total
- **Distribución**: 40% uppercase, 40% lowercase, 20% dígitos  
- **Patrones de posición**: Documentados para cada posición
- **Exclusiones**: Sin j, l, o, q, x, 0, 1

### Post-processing
- **Objetivo**: Corregir predicciones anómalas del ensemble
- **Método**: Detección de patrones garbage + correcciones directas
- **Estado**: Funcional pero necesita optimización
