#!/usr/bin/env python3
"""
Evaluador completo de todos los modelos contra 20 CAPTCHAs reales
Genera tabla comparativa de rendimiento
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import string
import argparse
import hashlib
from pathlib import Path
import json

# Configuración
FULL_CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
NUM_CLASSES = len(FULL_CHARSET)  # 62 clases
CAPTCHA_LENGTH = 6
HEIGHT = 80
WIDTH = 230

# Etiquetas conocidas para los 20 CAPTCHAs reales
KNOWN_LABELS = {
    "1.png": "UKhGh9",    "2.png": "26WanS",    "3.png": "e4TkHP",
    "4.png": "cGfFE2",    "5.png": "gnRYZe",    "6.png": "v76Ebu", 
    "7.png": "DUzp49",    "8.png": "MWR3mw",    "9.png": "3h2vUF",
    "10.png": "t2md2m",   "11.png": "UNKNOWN",  "12.png": "UNKNOWN",
    "13.png": "UNKNOWN",  "14.png": "UNKNOWN",  "15.png": "UNKNOWN",
    "16.png": "UNKNOWN",  "17.png": "UNKNOWN",  "18.png": "UNKNOWN",
    "19.png": "UNKNOWN",  "20.png": "UNKNOWN"
}

class FastCaptchaModel(nn.Module):
    """Modelo standalone con ResNet-18"""
    
    def __init__(self, num_classes=NUM_CLASSES, captcha_length=CAPTCHA_LENGTH):
        super().__init__()
        
        import torchvision.models as models
        self.backbone = models.resnet18(weights=None)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes * captcha_length)
        
        self.num_classes = num_classes
        self.captcha_length = captcha_length
        
    def forward(self, x):
        features = self.backbone(x)
        batch_size = features.size(0)
        outputs = features.view(batch_size, self.captcha_length, self.num_classes)
        return outputs

class LegacyCaptchaModel(nn.Module):
    """Modelo legacy con 36 clases"""
    
    def __init__(self):
        super().__init__()
        
        import torchvision.models as models
        self.backbone = models.resnet18(weights=None)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 36 * CAPTCHA_LENGTH)  # Solo 36 clases
        
    def forward(self, x):
        features = self.backbone(x)
        batch_size = features.size(0)
        outputs = features.view(batch_size, CAPTCHA_LENGTH, 36)
        return outputs

def get_model_info(model_path):
    """Extraer información del modelo"""
    try:
        # Verificar que el archivo no esté corrupto
        if os.path.getsize(model_path) < 1000:  # Archivos muy pequeños probablemente corruptos
            return {
                'path': model_path,
                'error': 'File too small (likely corrupted)',
                'size_mb': os.path.getsize(model_path) / (1024*1024)
            }
            
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        info = {
            'path': model_path,
            'size_mb': os.path.getsize(model_path) / (1024*1024),
            'hash': hashlib.md5(open(model_path, 'rb').read()).hexdigest()[:8]
        }
        
        if isinstance(checkpoint, dict):
            info.update({
                'epoch': checkpoint.get('epoch', 'N/A'),
                'best_acc': checkpoint.get('best_exact_acc', checkpoint.get('val_acc', 'N/A')),
                'char_acc': checkpoint.get('char_acc', 'N/A'),
                'val_loss': checkpoint.get('val_loss', 'N/A')
            })
            
            # Determinar número de clases
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # Buscar la capa FC final
            fc_keys = [k for k in state_dict.keys() if 'fc.weight' in k or 'classifier' in k]
            
            if fc_keys:
                fc_weight = state_dict[fc_keys[0]]
                total_outputs = fc_weight.shape[0]
                num_classes = total_outputs // CAPTCHA_LENGTH
                info['num_classes'] = num_classes
                
                if num_classes == 36:
                    info['vocabulary'] = "36 (0-9, a-z)"
                elif num_classes == 62:
                    info['vocabulary'] = "62 (0-9, a-z, A-Z)"
                else:
                    info['vocabulary'] = f"{num_classes} clases"
            else:
                info['num_classes'] = 'Unknown'
                info['vocabulary'] = 'Unknown'
                
        return info
        
    except Exception as e:
        return {
            'path': model_path,
            'error': str(e),
            'size_mb': os.path.getsize(model_path) / (1024*1024) if os.path.exists(model_path) else 0
        }

def load_model(model_path, device):
    """Cargar modelo según su tipo"""
    try:
        # Verificar tamaño del archivo
        if os.path.getsize(model_path) < 1000:
            return None, None, None
            
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Determinar tipo de modelo
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Determinar número de clases
        fc_keys = [k for k in state_dict.keys() if 'fc.weight' in k or 'classifier' in k]
        
        if fc_keys:
            fc_weight = state_dict[fc_keys[0]]
            total_outputs = fc_weight.shape[0]
            num_classes = total_outputs // CAPTCHA_LENGTH
            
            if num_classes == 36:
                # Modelo legacy con 36 clases
                model = LegacyCaptchaModel().to(device)
                model.load_state_dict(state_dict)
                charset = string.digits + string.ascii_lowercase  # Solo 36 caracteres
            else:
                # Modelo nuevo con 62 clases
                model = FastCaptchaModel(num_classes, CAPTCHA_LENGTH).to(device)
                model.load_state_dict(state_dict)
                charset = FULL_CHARSET  # 62 caracteres
                
            model.eval()
            return model, charset, num_classes
            
    except Exception as e:
        print(f"Error cargando {model_path}: {e}")
        return None, None, None

def evaluate_model(model, charset, model_info, captcha_dir, device):
    """Evaluar un modelo con los CAPTCHAs reales"""
    
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Obtener archivos de CAPTCHAs
    captcha_files = [f for f in os.listdir(captcha_dir) if f.endswith('.png')]
    captcha_files.sort(key=lambda x: int(x.split('.')[0]))
    
    results = {
        'model_info': model_info,
        'predictions': {},
        'perfect_matches': 0,
        'known_captchas': 0,
        'total_char_accuracy': 0,
        'avg_confidence': 0,
        'errors': []
    }
    
    total_confidence = 0
    
    with torch.no_grad():
        for filename in captcha_files:
            try:
                filepath = os.path.join(captcha_dir, filename)
                
                # Cargar imagen
                image = Image.open(filepath).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Predicción
                outputs = model(image_tensor)  # [1, 6, num_classes]
                probs = torch.softmax(outputs[0], dim=1)  # [6, num_classes]
                predicted_indices = torch.argmax(outputs[0], dim=1)  # [6]
                
                # Convertir a texto
                predicted_text = ''.join([charset[idx.item()] for idx in predicted_indices])
                
                # Calcular confianza
                confidences = [probs[i, predicted_indices[i]].item() for i in range(6)]
                avg_conf = sum(confidences) / len(confidences)
                total_confidence += avg_conf
                
                # Comparar con etiqueta conocida
                true_label = KNOWN_LABELS.get(filename, "UNKNOWN")
                
                result = {
                    'predicted': predicted_text,
                    'confidence': avg_conf,
                    'true_label': true_label
                }
                
                if true_label != "UNKNOWN":
                    # Calcular precisión por carácter
                    char_accuracy = sum(1 for p, t in zip(predicted_text, true_label) if p == t) / len(true_label)
                    result['char_accuracy'] = char_accuracy
                    results['total_char_accuracy'] += char_accuracy
                    results['known_captchas'] += 1
                    
                    if predicted_text == true_label:
                        results['perfect_matches'] += 1
                        result['perfect_match'] = True
                    else:
                        result['perfect_match'] = False
                
                results['predictions'][filename] = result
                
            except Exception as e:
                results['errors'].append(f"{filename}: {str(e)}")
    
    # Calcular estadísticas finales
    if results['known_captchas'] > 0:
        results['avg_char_accuracy'] = results['total_char_accuracy'] / results['known_captchas']
        results['perfect_match_rate'] = results['perfect_matches'] / results['known_captchas']
    
    results['avg_confidence'] = total_confidence / len(captcha_files)
    
    return results

def find_all_models():
    """Encontrar todos los modelos disponibles"""
    model_paths = []
    
    # Buscar en diferentes directorios
    search_dirs = [
        'models',
        'legacy/checkpoints_sssalud',
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith(('.pth', '.ckpt', '.pt')):
                        full_path = os.path.join(root, file)
                        # Filtrar archivos muy pequeños (probablemente corruptos)
                        if os.path.getsize(full_path) > 10000:  # > 10KB
                            model_paths.append(full_path)
    
    return sorted(set(model_paths))

def generate_comparison_table(all_results):
    """Generar tabla comparativa de todos los modelos"""
    
    print("\n" + "="*120)
    print("🎯 TABLA COMPARATIVA DE TODOS LOS MODELOS")
    print("="*120)
    
    # Header de tabla
    print("┌─────────────────────────────────────┬─────────┬────────┬──────────┬─────────┬─────────┬─────────┬──────────┐")
    print("│ Modelo                              │ Vocab   │ Época  │ Perfect  │ Char    │ Conf    │ Size    │ Status   │")
    print("├─────────────────────────────────────┼─────────┼────────┼──────────┼─────────┼─────────┼─────────┼──────────┤")
    
    # Ordenar por precisión exacta
    sorted_results = sorted(all_results, 
                           key=lambda x: x.get('perfect_match_rate', 0), 
                           reverse=True)
    
    for result in sorted_results:
        info = result['model_info']
        
        # Nombre del modelo (truncado)
        model_name = os.path.basename(info['path'])
        if len(model_name) > 35:
            model_name = model_name[:32] + "..."
        
        # Vocabulario
        vocab = info.get('vocabulary', 'Unknown')
        if len(vocab) > 7:
            vocab = vocab[:7]
        
        # Época
        epoch = str(info.get('epoch', 'N/A'))
        if len(epoch) > 6:
            epoch = epoch[:6]
            
        # Métricas
        if 'perfect_match_rate' in result:
            perfect = f"{result['perfect_match_rate']:.1%}"
            char_acc = f"{result['avg_char_accuracy']:.1%}"
            conf = f"{result['avg_confidence']:.3f}"
            
            # Status
            if result['perfect_match_rate'] > 0.8:
                status = "🎉 EXCELENTE"
            elif result['perfect_match_rate'] > 0.6:
                status = "✅ BUENO"
            elif result['perfect_match_rate'] > 0.3:
                status = "⚠️  REGULAR"
            else:
                status = "❌ POBRE"
        else:
            perfect = "ERROR"
            char_acc = "ERROR"
            conf = "ERROR"
            status = "❌ FALLO"
        
        size_mb = f"{info.get('size_mb', 0):.1f}MB"
        
        print(f"│ {model_name:<35} │ {vocab:<7} │ {epoch:<6} │ {perfect:<8} │ {char_acc:<7} │ {conf:<7} │ {size_mb:<7} │ {status:<8} │")
    
    print("└─────────────────────────────────────┴─────────┴────────┴──────────┴─────────┴─────────┴─────────┴──────────┘")
    
    # Estadísticas generales
    successful_models = [r for r in all_results if 'perfect_match_rate' in r]
    
    if successful_models:
        best_model = max(successful_models, key=lambda x: x['perfect_match_rate'])
        
        print(f"\n🏆 MEJOR MODELO:")
        print(f"   Archivo: {best_model['model_info']['path']}")
        print(f"   Precisión exacta: {best_model['perfect_match_rate']:.1%}")
        print(f"   Precisión por carácter: {best_model['avg_char_accuracy']:.1%}")
        print(f"   Vocabulario: {best_model['model_info'].get('vocabulary', 'Unknown')}")
        
        # Mostrar predicciones del mejor modelo
        print(f"\n📋 PREDICCIONES DEL MEJOR MODELO:")
        print("┌──────┬──────────┬──────────┬─────────┬────────┐")
        print("│ File │ Pred     │ Real     │ Conf    │ Match  │")
        print("├──────┼──────────┼──────────┼─────────┼────────┤")
        
        for filename in sorted(best_model['predictions'].keys(), key=lambda x: int(x.split('.')[0])):
            pred_info = best_model['predictions'][filename]
            
            pred = pred_info['predicted']
            real = pred_info['true_label'] if pred_info['true_label'] != "UNKNOWN" else "?"
            conf = f"{pred_info['confidence']:.3f}"
            match = "✅" if pred_info.get('perfect_match', False) else "❌" if real != "?" else "❓"
            
            print(f"│ {filename:<4} │ {pred:<8} │ {real:<8} │ {conf:<7} │ {match:<6} │")
        
        print("└──────┴──────────┴──────────┴─────────┴────────┘")

def main():
    parser = argparse.ArgumentParser(description='Evaluar todos los modelos contra CAPTCHAs reales')
    parser.add_argument('--captcha_dir', type=str, default='my_captchas', help='Directorio con CAPTCHAs reales')
    parser.add_argument('--output', type=str, help='Archivo JSON para guardar resultados')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.captcha_dir):
        print(f"❌ Directorio no encontrado: {args.captcha_dir}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Dispositivo: {device}")
    
    # Encontrar todos los modelos
    model_paths = find_all_models()
    print(f"🔍 Encontrados {len(model_paths)} archivos de modelo")
    
    if not model_paths:
        print("❌ No se encontraron modelos")
        return
    
    all_results = []
    
    # Evaluar cada modelo
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n📊 Evaluando modelo {i}/{len(model_paths)}: {os.path.basename(model_path)}")
        
        # Obtener información del modelo
        model_info = get_model_info(model_path)
        
        if 'error' in model_info:
            print(f"   ❌ Error: {model_info['error']}")
            all_results.append({'model_info': model_info, 'error': model_info['error']})
            continue
        
        print(f"   📋 Info: {model_info.get('vocabulary', 'Unknown')}, Época: {model_info.get('epoch', 'N/A')}")
        
        # Cargar y evaluar modelo
        model_result = load_model(model_path, device)
        
        if model_result is None or model_result[0] is None:
            print(f"   ❌ No se pudo cargar el modelo")
            all_results.append({'model_info': model_info, 'error': 'Failed to load'})
            continue
        
        model, charset, num_classes = model_result
        
        # Evaluar
        results = evaluate_model(model, charset, model_info, args.captcha_dir, device)
        
        if results['known_captchas'] > 0:
            print(f"   ✅ Perfectos: {results['perfect_matches']}/{results['known_captchas']} "
                  f"({results['perfect_match_rate']:.1%}), "
                  f"Char: {results['avg_char_accuracy']:.1%}")
        else:
            print(f"   ⚠️  No hay CAPTCHAs conocidos para evaluar")
        
        all_results.append(results)
        
        # Limpiar memoria
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Generar tabla comparativa
    generate_comparison_table(all_results)
    
    # Guardar resultados en JSON si se especifica
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Resultados guardados en: {args.output}")

if __name__ == "__main__":
    main()
