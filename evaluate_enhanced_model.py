#!/usr/bin/env python3
"""
Evaluador del modelo mejorado con CAPTCHAs reales
Prueba el modelo entrenado con 62 clases contra los 20 CAPTCHAs reales
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import string
from train_enhanced_model import EnhancedCaptchaModel

# Configuración
FULL_CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
REAL_CAPTCHAS_DIR = "my_captchas"

# Etiquetas conocidas de los CAPTCHAs reales (del documento)
KNOWN_LABELS = {
    "1.png": "UKhGh9",
    "2.png": "26WanS", 
    "3.png": "e4TkHP",
    "4.png": "cGfFE2",
    "5.png": "gnRYZe",
    "6.png": "v76Ebu",
    "7.png": "DUzp49",
    "8.png": "MWR3mw",
    "9.png": "3h2vUF",
    "10.png": "t2md2m",
    # Para las demás podemos hacer predicción ciega
}

class CaptchaEvaluator:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((64, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def _load_model(self, model_path: str) -> EnhancedCaptchaModel:
        """Cargar modelo entrenado"""
        print(f"🤖 Cargando modelo desde: {model_path}")
        
        if model_path.endswith('.ckpt'):
            # Modelo de PyTorch Lightning
            model = EnhancedCaptchaModel.load_from_checkpoint(model_path)
        else:
            # Checkpoint manual
            checkpoint = torch.load(model_path, map_location='cpu')
            model = EnhancedCaptchaModel()
            model.load_state_dict(checkpoint['state_dict'])
        
        model.eval()
        print("✅ Modelo cargado correctamente")
        return model
    
    def predict_single(self, image_path: str) -> tuple:
        """Predecir un CAPTCHA individual"""
        # Cargar y procesar imagen
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Predicción
        with torch.no_grad():
            outputs = self.model(image_tensor)  # [1, 6, 62]
            predicted_indices = torch.argmax(outputs, dim=2)[0]  # [6]
        
        # Convertir a texto
        predicted_text = ""
        confidences = []
        
        for i, idx in enumerate(predicted_indices):
            char = FULL_CHARSET[idx.item()]
            predicted_text += char
            
            # Calcular confianza
            probs = torch.softmax(outputs[0, i, :], dim=0)
            confidence = probs[idx].item()
            confidences.append(confidence)
        
        avg_confidence = sum(confidences) / len(confidences)
        
        return predicted_text, avg_confidence, confidences
    
    def evaluate_real_captchas(self):
        """Evaluar todos los CAPTCHAs reales"""
        print("🎯 EVALUANDO CAPTCHAS REALES")
        print("=" * 50)
        
        if not os.path.exists(REAL_CAPTCHAS_DIR):
            print(f"❌ Directorio no encontrado: {REAL_CAPTCHAS_DIR}")
            return
        
        # Obtener lista de archivos
        captcha_files = [f for f in os.listdir(REAL_CAPTCHAS_DIR) if f.endswith('.png')]
        captcha_files.sort(key=lambda x: int(x.split('.')[0]))  # Ordenar numéricamente
        
        results = []
        perfect_matches = 0
        total_accuracy = 0
        
        print(f"📁 Encontrados {len(captcha_files)} CAPTCHAs")
        print("\n🔍 RESULTADOS DETALLADOS:")
        print("-" * 80)
        
        for filename in captcha_files:
            filepath = os.path.join(REAL_CAPTCHAS_DIR, filename)
            
            # Hacer predicción
            predicted, avg_conf, char_confs = self.predict_single(filepath)
            
            # Comparar con etiqueta conocida si existe
            true_label = KNOWN_LABELS.get(filename, "UNKNOWN")
            
            if true_label != "UNKNOWN":
                # Calcular precisión carácter por carácter
                char_accuracy = sum(1 for p, t in zip(predicted, true_label) if p == t) / len(true_label)
                total_accuracy += char_accuracy
                
                if predicted == true_label:
                    perfect_matches += 1
                    status = "✅ PERFECTO"
                else:
                    status = "❌ DIFERENTE"
            else:
                status = "❓ DESCONOCIDO"
                char_accuracy = None
            
            # Mostrar resultado
            conf_str = f"{avg_conf:.3f}"
            if true_label != "UNKNOWN":
                acc_str = f"{char_accuracy:.3f}" if char_accuracy is not None else "N/A"
                print(f"{filename:>8} | {predicted:>6} | {true_label:>6} | {conf_str:>5} | {acc_str:>5} | {status}")
            else:
                print(f"{filename:>8} | {predicted:>6} | {'?':>6} | {conf_str:>5} | {'?':>5} | {status}")
            
            # Mostrar confianzas por carácter si es baja
            if avg_conf < 0.8:
                conf_details = " ".join([f"{c:.2f}" for c in char_confs])
                print(f"         Confianzas por carácter: {conf_details}")
            
            results.append({
                'filename': filename,
                'predicted': predicted,
                'true_label': true_label,
                'confidence': avg_conf,
                'accuracy': char_accuracy
            })
        
        # Estadísticas finales
        known_captchas = len([r for r in results if r['true_label'] != "UNKNOWN"])
        
        print("\n" + "=" * 50)
        print("📊 ESTADÍSTICAS FINALES:")
        
        if known_captchas > 0:
            avg_accuracy = total_accuracy / known_captchas
            perfect_rate = perfect_matches / known_captchas
            
            print(f"CAPTCHAs conocidos: {known_captchas}")
            print(f"Matches perfectos: {perfect_matches}/{known_captchas} ({perfect_rate:.1%})")
            print(f"Precisión promedio: {avg_accuracy:.1%}")
            
            # Comparar con resultados documentados
            print(f"\n📈 COMPARACIÓN CON MODELO ANTERIOR:")
            print(f"Modelo anterior (36 clases): 68.3% precisión, 2/10 perfectos")
            print(f"Modelo nuevo (62 clases): {avg_accuracy:.1%} precisión, {perfect_matches}/{known_captchas} perfectos")
            
            if avg_accuracy > 0.683:
                print("🎉 ¡MEJORA CONSEGUIDA!")
            elif avg_accuracy > 0.60:
                print("👍 Resultado prometedor")
            else:
                print("⚠️ Necesita más entrenamiento")
        
        return results
    
    def analyze_predictions(self, results):
        """Análizar patrones en las predicciones"""
        print(f"\n🔍 ANÁLISIS DE PATRONES:")
        
        # Caracteres más difíciles de predecir
        char_errors = {}
        for result in results:
            if result['true_label'] != "UNKNOWN":
                pred = result['predicted']
                true = result['true_label']
                
                for i, (p, t) in enumerate(zip(pred, true)):
                    if p != t:
                        if t not in char_errors:
                            char_errors[t] = 0
                        char_errors[t] += 1
        
        if char_errors:
            print("Caracteres más problemáticos:")
            sorted_errors = sorted(char_errors.items(), key=lambda x: x[1], reverse=True)
            for char, count in sorted_errors[:5]:
                print(f"  '{char}': {count} errores")

def main():
    """Función principal"""
    # Buscar modelo más reciente
    model_dir = "models/04_enhanced_62classes"
    
    if not os.path.exists(model_dir):
        print(f"❌ Directorio de modelos no encontrado: {model_dir}")
        print("Entrena primero el modelo con train_enhanced_model.py")
        return
    
    # Buscar checkpoint más reciente
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
    
    if not checkpoints:
        print(f"❌ No se encontraron checkpoints en {model_dir}")
        return
    
    # Usar el checkpoint con mayor precisión
    best_checkpoint = max(checkpoints, key=lambda x: float(x.split('val_acc=')[1].split('.ckpt')[0]))
    model_path = os.path.join(model_dir, best_checkpoint)
    
    print(f"🎯 Usando modelo: {best_checkpoint}")
    
    # Evaluar
    evaluator = CaptchaEvaluator(model_path)
    results = evaluator.evaluate_real_captchas()
    evaluator.analyze_predictions(results)

if __name__ == "__main__":
    main()
