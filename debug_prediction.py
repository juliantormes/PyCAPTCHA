#!/usr/bin/env python3
import torch
from model.model import captcha_model, model_resnet
from data.dataset import lst_to_str, CHARS
from PIL import Image
import torchvision.transforms as transforms

print("🔍 Debugging predicción paso a paso...")

# 1. Verificar el vocabulario
print(f"Vocabulario CHARS: {CHARS}")
print(f"Longitud: {len(CHARS)}")
print()

# 2. Cargar modelo
model = captcha_model(model_resnet())
checkpoint = torch.load('models/03_specialized_sssalud/model.pth', map_location='cpu')

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)
model.eval()

print("✅ Modelo cargado")

# 3. Probar con una imagen
image_path = 'my_captchas/1.png'  # UKhGh9
if torch.cuda.is_available():
    print("🔥 CUDA disponible")
else:
    print("💻 CPU solamente")

image = Image.open(image_path).convert('RGB')
print(f"Imagen: {image.size}")

# 4. Transformaciones
transform = transforms.Compose([
    transforms.Resize((60, 160)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image_tensor = transform(image).unsqueeze(0)
print(f"Tensor shape: {image_tensor.shape}")
print(f"Tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")

# 5. Predicción cruda
with torch.no_grad():
    output = model(image_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output type: {type(output)}")
    
    # Ver los logits crudos para la primera posición
    print(f"Primeros logits pos 0: {output[0, 0, :10]}")
    
    # Predicción con argmax
    prediction = torch.argmax(output, dim=-1)
    print(f"Prediction tensor: {prediction}")
    print(f"Prediction shape: {prediction.shape}")
    
    # Convertir a string
    pred_str = lst_to_str(prediction[0])
    print(f"Predicción final: '{pred_str}'")
    print(f"Real esperado: 'UKhGh9'")
    
    # Verificar cada posición
    print("\nDesglose por posición:")
    for i in range(6):
        idx = prediction[0][i].item()
        char = CHARS[idx] if idx < len(CHARS) else '?'
        confidence = torch.softmax(output[0, i, :], dim=0)[idx].item()
        print(f"Pos {i}: idx={idx:2d} -> '{char}' (conf: {confidence:.3f})")
