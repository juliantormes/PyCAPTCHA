#!/usr/bin/env python3
"""
Evaluador específico para modelos legacy con arquitectura original
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import string
import sys

# Añadir el path para importar el modelo original
sys.path.append('.')

try:
    from model.model import captcha_model, model_resnet
    LEGACY_MODEL_AVAILABLE = True
except ImportError:
    print("⚠️  No se pudo importar el modelo legacy original")
    LEGACY_MODEL_AVAILABLE = False

# Configuración
LEGACY_CHARSET = string.digits + string.ascii_lowercase  # Solo 36 caracteres
FULL_CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
CAPTCHA_LENGTH = 6
HEIGHT = 80
WIDTH = 230

# Etiquetas conocidas
KNOWN_LABELS = {
    "1.png": "UKhGh9",    "2.png": "26WanS",    "3.png": "e4TkHP",
    "4.png": "cGfFE2",    "5.png": "gnRYZe",    "6.png": "v76Ebu", 
    "7.png": "DUzp49",    "8.png": "MWR3mw",    "9.png": "3h2vUF",
    "10.png": "t2md2m"
}

def evaluate_legacy_model(model_path, captcha_dir="my_captchas"):
    """Evaluar modelo legacy con arquitectura original"""
    
    if not LEGACY_MODEL_AVAILABLE:
        print("❌ No se puede evaluar modelo legacy - falta importación")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Dispositivo: {device}")
    
    # Cargar modelo legacy
    print(f"🤖 Cargando modelo legacy: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Crear modelo con arquitectura original
        backbone = model_resnet()
        model = captcha_model(backbone)
        model.to(device)
        
        # Cargar pesos
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        
        epoch = checkpoint.get('epoch', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
        print(f"   Época: {epoch}")
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Evaluar CAPTCHAs
    if not os.path.exists(captcha_dir):
        print(f"❌ Directorio no encontrado: {captcha_dir}")
        return
    
    captcha_files = [f for f in os.listdir(captcha_dir) if f.endswith('.png')]
    captcha_files.sort(key=lambda x: int(x.split('.')[0]))
    
    print(f"🎯 Evaluando {len(captcha_files)} CAPTCHAs reales con modelo legacy...")
    print("=" * 70)
    
    # Header
    print("┌──────┬──────────┬──────────┬─────────┬────────┬─────────────┐")
    print("│ File │ Pred     │ Real     │ Conf    │ Status │ Observación │")
    print("├──────┼──────────┼──────────┼─────────┼────────┼─────────────┤")
    
    perfect_matches = 0
    total_char_accuracy = 0
    known_count = 0
    uppercase_failures = 0
    
    with torch.no_grad():
        for filename in captcha_files:
            filepath = os.path.join(captcha_dir, filename)
            
            # Cargar y procesar imagen
            image = Image.open(filepath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predicción
            outputs = model(image_tensor)  # [1, 6, 36]
            probs = torch.softmax(outputs[0], dim=1)  # [6, 36]
            predicted_indices = torch.argmax(outputs[0], dim=1)  # [6]
            
            # Convertir a texto (solo 36 caracteres disponibles)
            predicted_text = ''.join([LEGACY_CHARSET[idx.item()] for idx in predicted_indices])
            
            # Calcular confianza promedio
            confidences = [probs[i, predicted_indices[i]].item() for i in range(6)]
            avg_confidence = sum(confidences) / len(confidences)
            
            # Comparar con etiqueta conocida
            true_label = KNOWN_LABELS.get(filename, "UNKNOWN")
            
            if true_label != "UNKNOWN":
                # Verificar si tiene mayúsculas
                has_uppercase = any(c.isupper() for c in true_label)
                
                # Calcular precisión por carácter (solo minúsculas)
                char_accuracy = sum(1 for p, t in zip(predicted_text, true_label.lower()) if p == t) / len(true_label)
                total_char_accuracy += char_accuracy
                known_count += 1
                
                if predicted_text == true_label.lower():
                    if has_uppercase:
                        status = "⚠️ MAYÚS"
                        observation = "Correcto pero sin mayúsculas"
                        uppercase_failures += 1
                    else:
                        perfect_matches += 1
                        status = "✅ PERFECT"
                        observation = "Perfecto"
                elif predicted_text == true_label:
                    perfect_matches += 1
                    status = "✅ PERFECT"
                    observation = "Perfecto"
                else:
                    status = "❌ DIFF"
                    if has_uppercase:
                        observation = "Falló + mayúsculas"
                        uppercase_failures += 1
                    else:
                        observation = "Falló"
                
                true_display = true_label
            else:
                status = "❓ UNKNOWN"
                true_display = "?"
                observation = "Sin etiqueta"
            
            # Mostrar resultado
            print(f"│ {filename:<4} │ {predicted_text:<8} │ {true_display:<8} │ {avg_confidence:.3f}   │ {status:<6} │ {observation:<11} │")
    
    # Footer
    print("└──────┴──────────┴──────────┴─────────┴────────┴─────────────┘")
    
    # Estadísticas finales
    if known_count > 0:
        avg_accuracy = total_char_accuracy / known_count
        perfect_rate = perfect_matches / known_count
        
        print(f"\n🎉 RESULTADOS MODELO LEGACY (36 clases):")
        print(f"   CAPTCHAs conocidos: {known_count}")
        print(f"   Matches perfectos: {perfect_matches}/{known_count} ({perfect_rate:.1%})")
        print(f"   Precisión promedio: {avg_accuracy:.1%}")
        print(f"   Fallos por mayúsculas: {uppercase_failures}")
        
        # Mostrar limitación del vocabulario
        print(f"\n⚠️  LIMITACIÓN DETECTADA:")
        print(f"   Modelo legacy solo conoce: {len(LEGACY_CHARSET)} caracteres (0-9, a-z)")
        print(f"   CAPTCHAs reales usan: {len(FULL_CHARSET)} caracteres (0-9, a-z, A-Z)")
        print(f"   Mayúsculas perdidas: {uppercase_failures}/{known_count} CAPTCHAs afectados")
        
        # Proyección de mejora con 62 clases
        projected_improvement = perfect_matches + (uppercase_failures * 0.8)  # 80% de éxito en mayúsculas
        projected_rate = projected_improvement / known_count
        
        print(f"\n📈 PROYECCIÓN CON 62 CLASES:")
        print(f"   Precisión proyectada: {projected_rate:.1%}")
        print(f"   Mejora esperada: +{(projected_rate - perfect_rate)*100:.1f} puntos porcentuales")

def main():
    model_paths = [
        'legacy/checkpoints_sssalud/model.pth',
        'models/03_specialized_sssalud/model.pth',
        'models/02_advanced/model.pth',
        'models/01_baseline/model.pth'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"\n🔍 EVALUANDO: {model_path}")
            evaluate_legacy_model(model_path)
            break
    else:
        print("❌ No se encontró ningún modelo legacy")

if __name__ == "__main__":
    main()
