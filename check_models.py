#!/usr/bin/env python3
import torch
import os
import hashlib
from model.model import captcha_model, model_resnet

print("🔍 ANÁLISIS PROFUNDO DE MODELOS...")

current = 'models/03_specialized_sssalud/model.pth'
legacy = 'legacy/checkpoints_sssalud/model.pth'

if os.path.exists(current) and os.path.exists(legacy):
    print(f"📁 Current: {current}")
    print(f"📁 Legacy:  {legacy}")
    
    # Comparar hashes
    with open(current, 'rb') as f:
        hash1 = hashlib.md5(f.read()).hexdigest()
    with open(legacy, 'rb') as f:
        hash2 = hashlib.md5(f.read()).hexdigest()
    
    print(f"\n🔐 HASH COMPARISON:")
    print(f"Current: {hash1}")
    print(f"Legacy:  {hash2}")
    
    if hash1 == hash2:
        print("✅ IDÉNTICOS - El problema está en el código de predicción")
    else:
        print("❌ DIFERENTES - Los modelos son distintos")
        
    # Comparar metadata y arquitectura
    try:
        print(f"\n📊 ANÁLISIS DETALLADO:")
        c1 = torch.load(current, map_location='cpu')
        c2 = torch.load(legacy, map_location='cpu')
        
        # Información básica
        print(f"\n📋 METADATA:")
        if isinstance(c1, dict):
            print(f"Current - Epoch: {c1.get('epoch', 'N/A')}, Step: {c1.get('global_step', 'N/A')}")
        if isinstance(c2, dict):
            print(f"Legacy  - Epoch: {c2.get('epoch', 'N/A')}, Step: {c2.get('global_step', 'N/A')}")
        
        # Analizar arquitectura del modelo
        print(f"\n🏗️ ARQUITECTURA:")
        
        # Cargar modelos para análisis
        backbone1 = model_resnet()
        model1 = captcha_model(backbone1)
        
        backbone2 = model_resnet()
        model2 = captcha_model(backbone2)
        
        try:
            model1.load_state_dict(c1['state_dict'])
            fc1 = model1.model.resnet.fc
            print(f"Current FC: in={fc1.in_features}, out={fc1.out_features}")
            
            # Calcular clases por posición
            classes_per_pos1 = fc1.out_features // 6
            print(f"Current - Clases por posición: {classes_per_pos1}")
            
            if classes_per_pos1 == 36:
                print("Current - ⚠️  Solo minúsculas (0-9, a-z)")
            elif classes_per_pos1 == 62:
                print("Current - ✅ Vocabulario completo (0-9, a-z, A-Z)")
            else:
                print(f"Current - ❓ Vocabulario extraño: {classes_per_pos1} clases")
                
        except Exception as e:
            print(f"Error cargando current: {e}")
        
        try:
            model2.load_state_dict(c2['state_dict'])
            fc2 = model2.model.resnet.fc
            print(f"Legacy  FC: in={fc2.in_features}, out={fc2.out_features}")
            
            # Calcular clases por posición
            classes_per_pos2 = fc2.out_features // 6
            print(f"Legacy  - Clases por posición: {classes_per_pos2}")
            
            if classes_per_pos2 == 36:
                print("Legacy  - ⚠️  Solo minúsculas (0-9, a-z)")
            elif classes_per_pos2 == 62:
                print("Legacy  - ✅ Vocabulario completo (0-9, a-z, A-Z)")
            else:
                print(f"Legacy  - ❓ Vocabulario extraño: {classes_per_pos2} clases")
                
        except Exception as e:
            print(f"Error cargando legacy: {e}")
        
        # Comparar pesos específicos si son diferentes
        if hash1 != hash2:
            print(f"\n⚖️ COMPARANDO PESOS:")
            try:
                # Comparar solo algunos pesos clave
                state1 = c1['state_dict']
                state2 = c2['state_dict']
                
                # Verificar si tienen las mismas claves
                keys1 = set(state1.keys())
                keys2 = set(state2.keys())
                
                if keys1 == keys2:
                    print("✅ Mismas capas en ambos modelos")
                else:
                    print("❌ Diferentes capas entre modelos")
                    only_in_1 = keys1 - keys2
                    only_in_2 = keys2 - keys1
                    if only_in_1:
                        print(f"Solo en current: {list(only_in_1)[:3]}...")
                    if only_in_2:
                        print(f"Solo en legacy: {list(only_in_2)[:3]}...")
                
                # Comparar pesos de la capa FC
                fc_weight_key = 'model.resnet.fc.weight'
                if fc_weight_key in state1 and fc_weight_key in state2:
                    w1 = state1[fc_weight_key]
                    w2 = state2[fc_weight_key]
                    
                    if w1.shape == w2.shape:
                        diff = torch.abs(w1 - w2).mean()
                        print(f"FC weights diff: {diff:.6f}")
                        if diff < 1e-6:
                            print("✅ Pesos de FC prácticamente idénticos")
                        else:
                            print("❌ Pesos de FC significativamente diferentes")
                    else:
                        print(f"❌ FC shapes diferentes: {w1.shape} vs {w2.shape}")
                        
            except Exception as e:
                print(f"Error comparando pesos: {e}")
                
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        
else:
    print("❌ Archivos no encontrados")
    if not os.path.exists(current):
        print(f"   Falta: {current}")
    if not os.path.exists(legacy):
        print(f"   Falta: {legacy}")
