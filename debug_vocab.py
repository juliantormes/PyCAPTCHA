#!/usr/bin/env python3
import torch
from model.model import captcha_model, model_resnet
from data.dataset import lst_to_str
from PIL import Image
import torchvision.transforms as transforms

print("🔍 Debugging vocabulario...")

# Reconstruir el vocabulario basado en lst_to_str
def get_vocabulary():
    vocab = []
    for i in range(62):  # 0-9, a-z, A-Z
        if i < 10:
            vocab.append(chr(i + ord('0')))  # 0-9
        elif i < 36:
            vocab.append(chr(i + ord('a') - 10))  # a-z
        else:
            vocab.append(chr(i + ord('A') - 36))  # A-Z
    return vocab

vocab = get_vocabulary()
print(f"Vocabulario completo: {''.join(vocab)}")
print(f"Longitud: {len(vocab)}")
print()

# Cargar modelo y hacer predicción
model = captcha_model(model_resnet())
checkpoint = torch.load('models/03_specialized_sssalud/model.pth', map_location='cpu')
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
model.load_state_dict(state_dict, strict=False)
model.eval()

# Probar con imagen
image = Image.open('my_captchas/1.png').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((60, 160)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=-1)
    pred_str = lst_to_str(prediction[0])
    
    print(f"Predicción: '{pred_str}'")
    print(f"Expected:   'UKhGh9'")
    print()
    
    print("Análisis por posición:")
    for i in range(6):
        idx = prediction[0][i].item()
        char = vocab[idx] if idx < len(vocab) else '?'
        probs = torch.softmax(output[0, i, :], dim=0)
        top5_indices = torch.topk(probs, 5).indices
        top5_chars = [vocab[j.item()] for j in top5_indices]
        top5_probs = [probs[j].item() for j in top5_indices]
        
        print(f"Pos {i}: '{char}' (idx={idx})")
        print(f"       Top 5: {list(zip(top5_chars, [f'{p:.3f}' for p in top5_probs]))}")
        print()
