#!/usr/bin/env python3
import torch
from model.model import captcha_model, model_resnet

def analyze_model_classes():
    """Analizar cuántas clases puede predecir realmente el modelo"""
    
    print("🔍 ANÁLISIS DE CLASES DEL MODELO")
    
    # Cargar modelo
    model_path = 'models/03_specialized_sssalud/model.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    
    backbone = model_resnet()
    model = captcha_model(backbone)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Verificar la capa final
    fc_layer = model.model.resnet.fc
    print(f"📊 Capa FC: {fc_layer}")
    print(f"📐 Input features: {fc_layer.in_features}")
    print(f"📤 Output features: {fc_layer.out_features}")
    
    # Calcular las dimensiones esperadas
    total_outputs = fc_layer.out_features
    positions = 6
    classes_per_position = total_outputs // positions
    
    print(f"\n🧮 ANÁLISIS DE DIMENSIONES:")
    print(f"  Total outputs: {total_outputs}")
    print(f"  Posiciones CAPTCHA: {positions}")
    print(f"  Clases por posición: {classes_per_position}")
    
    if classes_per_position == 62:
        print("✅ Modelo correcto: 62 clases (0-9, a-z, A-Z)")
    elif classes_per_position == 36:
        print("⚠️  Modelo limitado: 36 clases (probablemente solo 0-9, a-z)")
    else:
        print(f"❌ Modelo con dimensiones extrañas: {classes_per_position} clases")
    
    # Determinar qué vocabulario usa
    if classes_per_position == 36:
        vocab_36 = "0123456789abcdefghijklmnopqrstuvwxyz"
        print(f"\n📚 Vocabulario probable (36 chars): {vocab_36}")
        print(f"📏 Longitud: {len(vocab_36)}")
        
        # Verificar dónde están los caracteres problemáticos
        e_idx = vocab_36.index('e')
        z_idx = vocab_36.index('z')
        o_idx = vocab_36.index('o')
        
        print(f"\n🎯 Índices en vocabulario de 36:")
        print(f"  'e' está en índice: {e_idx}")
        print(f"  'z' está en índice: {z_idx}")
        print(f"  'o' está en índice: {o_idx}")
        
        return vocab_36
    
    elif classes_per_position == 62:
        vocab_62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        print(f"\n📚 Vocabulario completo (62 chars): {vocab_62}")
        return vocab_62
    
    return None

def test_prediction_with_correct_vocab():
    """Probar predicción con el vocabulario correcto"""
    
    vocab = analyze_model_classes()
    
    if vocab is None:
        print("❌ No se pudo determinar el vocabulario")
        return
    
    print(f"\n🧪 PROBANDO PREDICCIÓN CON VOCABULARIO CORRECTO")
    
    # Cargar modelo
    model_path = 'models/03_specialized_sssalud/model.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    
    backbone = model_resnet()
    model = captcha_model(backbone)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Cargar imagen
    from PIL import Image
    import torchvision.transforms as transforms
    
    image_path = 'assets/captcha.png'
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((64, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    # Predicción
    with torch.no_grad():
        output = model(image_tensor)
    
    # Aplicar softmax
    softmax = torch.nn.Softmax(dim=2)
    probs = softmax(output)
    
    print(f"\n🎯 PREDICCIÓN CON VOCABULARIO CORRECTO:")
    prediction = ""
    
    for pos in range(6):
        pos_probs = probs[0, pos, :]
        
        # Obtener top predicción
        top_prob, top_idx = torch.topk(pos_probs, 1)
        
        predicted_char = vocab[top_idx.item()]
        prediction += predicted_char
        
        print(f"  Pos {pos}: '{predicted_char}' (conf: {top_prob.item():.4f})")
    
    print(f"\n🎉 PREDICCIÓN CORREGIDA: '{prediction}'")
    
    # Comparar con el CAPTCHA real esperado
    expected = "UKhGh9"  # Del documento
    print(f"🎯 CAPTCHA real esperado: '{expected}'")
    
    if prediction == expected:
        print("✅ ¡PREDICCIÓN PERFECTA!")
    else:
        print("❌ Aún hay diferencias, pero ya no es 'zzeeeo'")
        
        # Analizar diferencias carácter por carácter
        print("\n🔍 Análisis carácter por carácter:")
        for i, (pred, real) in enumerate(zip(prediction, expected)):
            status = "✅" if pred == real else "❌"
            print(f"  Pos {i}: pred='{pred}' vs real='{real}' {status}")

if __name__ == '__main__':
    test_prediction_with_correct_vocab()
