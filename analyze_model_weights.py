#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from model.model import captcha_model as CaptchaModel, model_resnet

def analyze_model_weights():
    """Análisis detallado de los pesos del modelo para detectar problemas"""
    
    print('=== ANÁLISIS COMPLETO DEL MODELO ===')
    model_path = 'models/03_specialized_sssalud/model.pth'
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print('Información del checkpoint:')
    print(f'Epoch: {checkpoint.get("epoch", "N/A")}')
    print(f'Global step: {checkpoint.get("global_step", "N/A")}')
    if 'train_loss' in checkpoint:
        print(f'Train loss: {checkpoint["train_loss"]}')
    if 'lr_schedulers' in checkpoint:
        print(f'Learning rate schedulers: {len(checkpoint["lr_schedulers"])}')
    
    # Cargar modelo
    # Crear el modelo con la estructura correcta
    backbone = model_resnet()
    model = CaptchaModel(backbone)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Verificar arquitectura
    print(f'\nModelo: {model.model.__class__.__name__}')
    
    # Analizar clasificador final
    classifier = model.model.resnet.fc
    print(f'\nClasificador final: {classifier}')
    print(f'Input features: {classifier.in_features}')
    print(f'Output classes: {classifier.out_features}')
    
    # Analizar pesos y bias
    weights = classifier.weight.data
    bias = classifier.bias.data
    
    print(f'\nEstadísticas generales:')
    print(f'Weight shape: {weights.shape}')
    print(f'Weight mean: {weights.mean():.6f}, std: {weights.std():.6f}')
    print(f'Weight min: {weights.min():.6f}, max: {weights.max():.6f}')
    print(f'Bias mean: {bias.mean():.6f}, std: {bias.std():.6f}')
    print(f'Bias min: {bias.min():.6f}, max: {bias.max():.6f}')
    
    # Buscar patrones problemáticos
    weights_np = weights.numpy()
    bias_np = bias.numpy()
    
    print(f'\n=== ANÁLISIS POR POSICIÓN ===')
    # Cada 62 clases representan una posición del CAPTCHA
    vocab_size = 62  # 0-9 (10) + a-z (26) + A-Z (26)
    num_positions = weights_np.shape[0] // vocab_size
    
    for pos in range(num_positions):
        start_idx = pos * vocab_size
        end_idx = start_idx + vocab_size
        
        pos_weights = weights_np[start_idx:end_idx]
        pos_bias = bias_np[start_idx:end_idx]
        
        print(f'\nPosición {pos}:')
        print(f'  Weight mean: {pos_weights.mean():.6f}, std: {pos_weights.std():.6f}')
        print(f'  Bias mean: {pos_bias.mean():.6f}, std: {pos_bias.std():.6f}')
        
        # Encontrar los caracteres con bias más alto
        top_bias_indices = np.argsort(pos_bias)[-5:][::-1]
        
        # Mapear índices a caracteres
        vocab = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        print(f'  Top 5 bias más altos:')
        for i, idx in enumerate(top_bias_indices):
            char = vocab[idx]
            bias_val = pos_bias[idx]
            print(f'    {i+1}. "{char}" (idx {idx}): bias = {bias_val:.6f}')
    
    # Verificar si hay algún patrón extremo
    print(f'\n=== DETECCIÓN DE PROBLEMAS ===')
    
    # Buscar bias extremadamente altos
    extreme_bias = np.abs(bias_np) > 10
    if extreme_bias.any():
        print(f'¡ADVERTENCIA! {extreme_bias.sum()} clases tienen bias extremo (>10)')
        extreme_indices = np.where(extreme_bias)[0]
        for idx in extreme_indices[:10]:  # Mostrar solo los primeros 10
            pos = idx // vocab_size
            char_idx = idx % vocab_size
            char = vocab[char_idx]
            print(f'  Posición {pos}, carácter "{char}": bias = {bias_np[idx]:.6f}')
    
    # Buscar pesos extremos
    extreme_weights = np.abs(weights_np) > 5
    if extreme_weights.any():
        print(f'¡ADVERTENCIA! {extreme_weights.sum()} pesos son extremos (>5)')
    
    return weights_np, bias_np, vocab

if __name__ == '__main__':
    weights, bias, vocab = analyze_model_weights()
