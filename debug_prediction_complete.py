#!/usr/bin/env python3
import torch
import torch.nn as nn
from model.model import captcha_model, model_resnet
from data.dataset import lst_to_str, str_to_lst
from PIL import Image
import torchvision.transforms as transforms

print("🔍 Debugging predicción completa del pipeline...")

def create_vocab():
    """Crear el vocabulario basado en la función str_to_lst"""
    vocab = ""
    # Números 0-9 (índices 0-9)
    for i in range(10):
        vocab += str(i)
    # Letras minúsculas a-z (índices 10-35)
    for i in range(26):
        vocab += chr(ord('a') + i)
    # Letras mayúsculas A-Z (índices 36-61)
    for i in range(26):
        vocab += chr(ord('A') + i)
    return vocab

def test_complete_pipeline():
    """Análisis completo del pipeline de predicción"""
    
    # 1. Crear vocabulario y verificar
    CHARS = create_vocab()
    print(f"📚 Vocabulario CHARS: {CHARS}")
    print(f"📏 Longitud vocabulario: {len(CHARS)}")
    
    # Mapear algunos caracteres importantes
    z_idx = CHARS.index('z') if 'z' in CHARS else -1
    e_idx = CHARS.index('e') if 'e' in CHARS else -1
    o_idx = CHARS.index('o') if 'o' in CHARS else -1
    
    print(f"🎯 Índices problemáticos: z={z_idx}, e={e_idx}, o={o_idx}")
    
    # 2. Cargar modelo
    print("🤖 Cargando modelo...")
    model_path = 'models/03_specialized_sssalud/model.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    
    backbone = model_resnet()
    model = captcha_model(backbone)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(f"✅ Modelo cargado - Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # 3. Cargar y procesar imagen
    print("🖼️ Procesando imagen...")
    image_path = 'assets/captcha.png'
    image = Image.open(image_path).convert('RGB')
    
    # Usar las mismas transformaciones que el dataset original
    transform = transforms.Compose([
        transforms.Resize((64, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    print(f"📐 Tensor shape: {image_tensor.shape}")
    print(f"📊 Tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    # 4. Predicción
    print("🎯 Ejecutando predicción...")
    with torch.no_grad():
        output = model(image_tensor)
    
    print(f"📤 Output shape: {output.shape}")
    print(f"📈 Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # 5. Aplicar softmax y analizar probabilidades
    print("🧮 Analizando probabilidades...")
    
    # El output debería ser [batch_size, 6, 62]
    if len(output.shape) == 3 and output.shape[1:] == (6, 62):
        print("✅ Forma de output correcta")
    else:
        print(f"❌ Forma incorrecta: esperada (1, 6, 62), obtenida {output.shape}")
    
    softmax = torch.nn.Softmax(dim=2)
    probs = softmax(output)
    
    # 6. Decodificar predicción
    print("\n🔤 ANÁLISIS POR POSICIÓN:")
    prediction = ""
    
    for pos in range(6):
        pos_probs = probs[0, pos, :]
        
        # Obtener top 5 predicciones
        top_probs, top_indices = torch.topk(pos_probs, 5)
        
        predicted_idx = top_indices[0].item()
        predicted_char = CHARS[predicted_idx]
        prediction += predicted_char
        
        print(f"\n  Posición {pos}:")
        print(f"    🎯 Predicción: '{predicted_char}' (conf: {top_probs[0]:.4f})")
        print(f"    📋 Top 5:")
        
        for i in range(5):
            char = CHARS[top_indices[i].item()]
            prob = top_probs[i].item()
            print(f"      {i+1}. '{char}': {prob:.4f}")
        
        # Verificar si los caracteres problemáticos tienen alta probabilidad
        if z_idx >= 0:
            z_prob = pos_probs[z_idx].item()
            if z_prob > 0.1:  # Si z tiene más del 10% de probabilidad
                print(f"    ⚠️  'z' tiene probabilidad alta: {z_prob:.4f}")
        
        if e_idx >= 0:
            e_prob = pos_probs[e_idx].item()
            if e_prob > 0.1:
                print(f"    ⚠️  'e' tiene probabilidad alta: {e_prob:.4f}")
                
        if o_idx >= 0:
            o_prob = pos_probs[o_idx].item()
            if o_prob > 0.1:
                print(f"    ⚠️  'o' tiene probabilidad alta: {o_prob:.4f}")
    
    print(f"\n🎉 PREDICCIÓN FINAL: '{prediction}'")
    
    # 7. Comparar con la función lst_to_str del dataset
    print("\n🔍 Verificando función lst_to_str...")
    
    # Simular lo que hace lst_to_str
    pred_indices = []
    for pos in range(6):
        pos_probs = probs[0, pos, :]
        pred_idx = torch.argmax(pos_probs).item()
        pred_indices.append(pred_idx)
    
    lst_str_result = lst_to_str(pred_indices)
    print(f"📝 lst_to_str resultado: '{lst_str_result}'")
    
    if prediction == lst_str_result:
        print("✅ Ambos métodos coinciden")
    else:
        print("❌ Los métodos difieren - hay un problema en la decodificación")
    
    return prediction, lst_str_result

if __name__ == '__main__':
    pred1, pred2 = test_complete_pipeline()
