#!/usr/bin/env python3
"""
Quick test to verify all import problems are solved
"""
import os
import sys
import importlib.util

print("🔍 Testing all imports...")

def test_import(module_path, module_name, description):
    """Test if a module can be imported"""
    try:
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✅ {description}")
            return True
        else:
            print(f"❌ {description} - File not found: {module_path}")
            return False
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return False

# Test all problematic imports
tests = [
    ("scripts/training/advanced_augmentation.py", "advanced_augmentation", "Advanced Augmentation"),
    ("models/04_ensemble/ensemble_predictor.py", "ensemble_predictor", "Ensemble Predictor"),
    ("experiments/postprocessing/captcha_postprocessor.py", "captcha_postprocessor", "Captcha Postprocessor"),
    ("scripts/training/custom_losses.py", "custom_losses", "Custom Losses"),
]

print("\n" + "="*50)
success_count = 0
for module_path, module_name, description in tests:
    if test_import(module_path, module_name, description):
        success_count += 1

print(f"\n📊 Results: {success_count}/{len(tests)} imports successful")

if success_count == len(tests):
    print("🎉 All import problems have been resolved!")
else:
    print("⚠️ Some import problems remain")

# Test basic dependencies
print("\n🔍 Testing basic dependencies...")
try:
    import torch
    print("✅ PyTorch")
except:
    print("❌ PyTorch")

try:
    import cv2
    print("✅ OpenCV")
except:
    print("❌ OpenCV")

try:
    from PIL import Image
    print("✅ Pillow")
except:
    print("❌ Pillow")
