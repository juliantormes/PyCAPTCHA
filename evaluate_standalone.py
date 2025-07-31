#!/usr/bin/env python3
"""
Evaluador rápido para modelo standalone
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import string
import argparse

# Configuración
FULL_CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
NUM_CLASSES = len(FULL_CHARSET)  # 62 clases
CAPTCHA_LENGTH = 6
HEIGHT = 80
WIDTH = 230

# Etiquetas conocidas para validación
KNOWN_LABELS = {
    "1.png": "UKhGh9",    "2.png": "26WanS",    "3.png": "e4TkHP",
    "4.png": "cGfFE2",    "5.png": "gnRYZe",    "6.png": "v76Ebu", 
    "7.png": "DUzp49",    "8.png": "MWR3mw",    "9.png": "3h2vUF",
    "10.png": "t2md2m"
}

class FastCaptchaModel(nn.Module):
    """Mismo modelo que en entrenamiento"""
    
    def __init__(self, num_classes=NUM_CLASSES, captcha_length=CAPTCHA_LENGTH):
        super().__init__()
        
        import torchvision.models as models
        self.backbone = models.resnet18(weights=None)  # Sin pesos preentrenados al cargar
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes * captcha_length)
        
        self.num_classes = num_classes
        self.captcha_length = captcha_length
        
    def forward(self, x):
        features = self.backbone(x)
        batch_size = features.size(0)
        outputs = features.view(batch_size, self.captcha_length, self.num_classes)
        return outputs

def evaluate_real_captchas(model_path: str, captcha_dir: str = "my_captchas"):
    """Evaluar modelo con CAPTCHAs reales"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Dispositivo: {device}")
    
    # Cargar modelo
    print(f"🤖 Cargando modelo: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = FastCaptchaModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', 'N/A')
    best_acc = checkpoint.get('best_exact_acc', 'N/A')
    print(f"   Época: {epoch}, Mejor precisión: {best_acc:.3f}")
    
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
    
    print(f"🎯 Evaluando {len(captcha_files)} CAPTCHAs reales...")
    print("=" * 70)
    
    # Header
    print("┌──────┬──────────┬──────────┬─────────┬────────┐")
    print("│ File │ Pred     │ Real     │ Conf    │ Status │")
    print("├──────┼──────────┼──────────┼─────────┼────────┤")
    
    perfect_matches = 0
    total_char_accuracy = 0
    known_count = 0
    
    with torch.no_grad():
        for filename in captcha_files:
            filepath = os.path.join(captcha_dir, filename)
            
            # Cargar y procesar imagen
            image = Image.open(filepath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predicción
            outputs = model(image_tensor)  # [1, 6, 62]
            probs = torch.softmax(outputs[0], dim=1)  # [6, 62]
            predicted_indices = torch.argmax(outputs[0], dim=1)  # [6]
            
            # Convertir a texto
            predicted_text = ''.join([FULL_CHARSET[idx.item()] for idx in predicted_indices])
            
            # Calcular confianza promedio
            confidences = [probs[i, predicted_indices[i]].item() for i in range(6)]
            avg_confidence = sum(confidences) / len(confidences)
            
            # Comparar con etiqueta conocida
            true_label = KNOWN_LABELS.get(filename, "UNKNOWN")
            
            if true_label != "UNKNOWN":
                # Calcular precisión por carácter
                char_accuracy = sum(1 for p, t in zip(predicted_text, true_label) if p == t) / len(true_label)
                total_char_accuracy += char_accuracy
                known_count += 1
                
                if predicted_text == true_label:
                    perfect_matches += 1
                    status = "✅ PERFECT"
                else:
                    status = "❌ DIFF"
                
                true_display = true_label
            else:
                status = "❓ UNKNOWN"
                true_display = "?"
                char_accuracy = None
            
            # Mostrar resultado
            print(f"│ {filename:<4} │ {predicted_text:<8} │ {true_display:<8} │ {avg_confidence:.3f}   │ {status:<6} │")
    
    # Footer
    print("└──────┴──────────┴──────────┴─────────┴────────┘")
    
    # Estadísticas finales
    if known_count > 0:
        avg_accuracy = total_char_accuracy / known_count
        perfect_rate = perfect_matches / known_count
        
        print(f"\n🎉 RESULTADOS FINALES:")
        print(f"   CAPTCHAs conocidos: {known_count}")
        print(f"   Matches perfectos: {perfect_matches}/{known_count} ({perfect_rate:.1%})")
        print(f"   Precisión promedio: {avg_accuracy:.1%}")
        
        # Comparación con baseline
        baseline_acc = 0.683
        baseline_perfect = 2
        
        print(f"\n📊 COMPARACIÓN:")
        print(f"   Modelo anterior (36 clases): {baseline_acc:.1%} precisión, {baseline_perfect}/10 perfectos")
        print(f"   Modelo nuevo (62 clases):    {avg_accuracy:.1%} precisión, {perfect_matches}/{known_count} perfectos")
        
        if avg_accuracy > baseline_acc:
            improvement = (avg_accuracy - baseline_acc) * 100
            print(f"   🎉 ¡MEJORA DE +{improvement:.1f} puntos porcentuales!")
        elif avg_accuracy > 0.60:
            print(f"   👍 Resultado prometedor")
        else:
            print(f"   ⚠️  Necesita más entrenamiento")

def main():
    parser = argparse.ArgumentParser(description='Evaluar modelo con CAPTCHAs reales')
    parser.add_argument('model_path', type=str, help='Path al modelo .pth')
    parser.add_argument('--captcha_dir', type=str, default='my_captchas', help='Directorio con CAPTCHAs reales')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Modelo no encontrado: {args.model_path}")
        return
    
    evaluate_real_captchas(args.model_path, args.captcha_dir)

if __name__ == "__main__":
    main()
