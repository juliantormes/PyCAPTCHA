#!/usr/bin/env python3
"""
Evaluación rápida del modelo GPU con los 20 CAPTCHAs reales
Optimizado para RTX 3070
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import string
from train_fast_gpu import FastCaptchaModel

# Configuración
FULL_CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
REAL_CAPTCHAS_DIR = "my_captchas"
HEIGHT = 80
WIDTH = 230

# Etiquetas conocidas
KNOWN_LABELS = {
    "1.png": "UKhGh9",    "2.png": "26WanS",    "3.png": "e4TkHP",
    "4.png": "cGfFE2",    "5.png": "gnRYZe",    "6.png": "v76Ebu", 
    "7.png": "DUzp49",    "8.png": "MWR3mw",    "9.png": "3h2vUF",
    "10.png": "t2md2m"
}

class FastCaptchaEvaluator:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Usando: {self.device}")
        
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((HEIGHT, WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def _load_model(self, model_path: str) -> FastCaptchaModel:
        """Cargar modelo optimizado"""
        print(f"🤖 Cargando modelo desde: {model_path}")
        
        if model_path.endswith('.ckpt'):
            model = FastCaptchaModel.load_from_checkpoint(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = FastCaptchaModel()
            model.load_state_dict(checkpoint['state_dict'])
        
        model.to(self.device)
        model.eval()
        print("✅ Modelo cargado y optimizado para GPU")
        return model
    
    @torch.no_grad()  # Optimización: no calcular gradientes
    def predict_single(self, image_path: str) -> tuple:
        """Predicción optimizada individual"""
        # Cargar y procesar imagen
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predicción en GPU
        outputs = self.model(image_tensor)  # [1, 6, 62]
        
        # Procesar resultados
        predicted_indices = torch.argmax(outputs, dim=2)[0]  # [6]
        probs = torch.softmax(outputs[0], dim=1)  # [6, 62]
        
        # Convertir a texto y confianzas
        predicted_text = ""
        confidences = []
        
        for i, idx in enumerate(predicted_indices):
            char = FULL_CHARSET[idx.item()]
            predicted_text += char
            confidence = probs[i, idx].item()
            confidences.append(confidence)
        
        avg_confidence = sum(confidences) / len(confidences)
        return predicted_text, avg_confidence, confidences
    
    def evaluate_real_captchas(self):
        """Evaluación rápida de todos los CAPTCHAs"""
        print("🎯 EVALUACIÓN RÁPIDA CON GPU")
        print("=" * 60)
        
        if not os.path.exists(REAL_CAPTCHAS_DIR):
            print(f"❌ Directorio no encontrado: {REAL_CAPTCHAS_DIR}")
            return
        
        # Lista de archivos
        captcha_files = [f for f in os.listdir(REAL_CAPTCHAS_DIR) if f.endswith('.png')]
        captcha_files.sort(key=lambda x: int(x.split('.')[0]))
        
        print(f"📁 Evaluando {len(captcha_files)} CAPTCHAs")
        print(f"🚀 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print()
        
        # Header de tabla
        print("┌────────┬────────┬────────┬─────────┬─────────┬──────────────┐")
        print("│ File   │ Pred   │ Real   │ Conf    │ Acc     │ Status       │")
        print("├────────┼────────┼────────┼─────────┼─────────┼──────────────┤")
        
        results = []
        perfect_matches = 0
        total_accuracy = 0
        known_count = 0
        
        # Evaluar cada CAPTCHA
        for filename in captcha_files:
            filepath = os.path.join(REAL_CAPTCHAS_DIR, filename)
            
            # Predicción
            predicted, avg_conf, char_confs = self.predict_single(filepath)
            
            # Comparar con etiqueta conocida
            true_label = KNOWN_LABELS.get(filename, "UNKNOWN")
            
            if true_label != "UNKNOWN":
                # Calcular precisión
                char_accuracy = sum(1 for p, t in zip(predicted, true_label) if p == t) / len(true_label)
                total_accuracy += char_accuracy
                known_count += 1
                
                if predicted == true_label:
                    perfect_matches += 1
                    status = "✅ PERFECTO"
                else:
                    status = "❌ DIFERENTE"
                
                acc_str = f"{char_accuracy:.3f}"
            else:
                status = "❓ DESCONOCIDO"
                acc_str = "N/A"
                char_accuracy = None
            
            # Mostrar resultado en tabla
            conf_str = f"{avg_conf:.3f}"
            true_str = true_label if true_label != "UNKNOWN" else "?"
            
            print(f"│ {filename:<6} │ {predicted:<6} │ {true_str:<6} │ {conf_str:<7} │ {acc_str:<7} │ {status:<12} │")
            
            results.append({
                'filename': filename,
                'predicted': predicted,
                'true_label': true_label,
                'confidence': avg_conf,
                'accuracy': char_accuracy
            })
        
        # Footer de tabla
        print("└────────┴────────┴────────┴─────────┴─────────┴──────────────┘")
        
        # Estadísticas finales
        if known_count > 0:
            avg_accuracy = total_accuracy / known_count
            perfect_rate = perfect_matches / known_count
            
            print(f"\n🎉 RESULTADOS FINALES:")
            print(f"   CAPTCHAs conocidos: {known_count}")
            print(f"   Matches perfectos: {perfect_matches}/{known_count} ({perfect_rate:.1%})")
            print(f"   Precisión promedio: {avg_accuracy:.1%}")
            
            # Comparación
            baseline_acc = 0.683  # Resultado documentado del modelo anterior
            baseline_perfect = 2   # Matches perfectos documentados
            
            print(f"\n📊 COMPARACIÓN:")
            print(f"   Modelo anterior (36 clases): {baseline_acc:.1%} precisión, {baseline_perfect}/10 perfectos")
            print(f"   Modelo nuevo (62 clases):    {avg_accuracy:.1%} precisión, {perfect_matches}/{known_count} perfectos")
            
            if avg_accuracy > baseline_acc:
                improvement = (avg_accuracy - baseline_acc) * 100
                print(f"   🎉 ¡MEJORA DE +{improvement:.1f} puntos porcentuales!")
            elif avg_accuracy > 0.60:
                print(f"   👍 Resultado prometedor, cerca del objetivo")
            else:
                print(f"   ⚠️  Necesita más entrenamiento")
            
            # Estadísticas de GPU
            if torch.cuda.is_available():
                print(f"\n💾 Memoria GPU usada: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        
        return results

def main():
    """Función principal de evaluación"""
    model_dir = "models/05_fast_gpu"
    
    if not os.path.exists(model_dir):
        print(f"❌ Directorio de modelos no encontrado: {model_dir}")
        print("Entrena primero el modelo con train_fast_gpu.py")
        return
    
    # Buscar mejor checkpoint
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
    
    if not checkpoints:
        print(f"❌ No se encontraron checkpoints en {model_dir}")
        return
    
    # Usar checkpoint con mayor precisión
    best_checkpoint = max(checkpoints, 
                         key=lambda x: float(x.split('val_acc=')[1].split('.ckpt')[0]) 
                         if 'val_acc=' in x else 0.0)
    
    model_path = os.path.join(model_dir, best_checkpoint)
    
    print(f"🎯 Evaluando modelo: {best_checkpoint}")
    
    # Evaluar
    evaluator = FastCaptchaEvaluator(model_path)
    results = evaluator.evaluate_real_captchas()

if __name__ == "__main__":
    main()
