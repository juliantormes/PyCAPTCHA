#!/usr/bin/env python3
"""
Script principal para probar todos los modelos de forma organizada
"""
import os
import sys
import importlib.util

# Get project root (2 levels up from scripts/testing)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

def test_model(model_name, model_path, test_images):
    """Test individual model"""
    print(f"\n🔍 Testing {model_name}")
    print("-" * 50)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        import torch
        from model.model import captcha_model, model_resnet
        from PIL import Image
        import torchvision.transforms as transforms
        from data.dataset import lst_to_str
        
        # Load model
        model = captcha_model(model_resnet())
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Preprocessing
        transform = transforms.Compose([
            transforms.Resize((60, 160)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Test images
        for img_path, real_value in test_images:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(image_tensor)
                    prediction = torch.argmax(output, dim=-1)
                    pred_str = lst_to_str(prediction[0])
                    
                match = "✅" if pred_str == real_value else "❌"
                print(f"{os.path.basename(img_path):12} | {pred_str:8} | {real_value:8} | {match}")
            
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")

def test_ensemble():
    """Test ensemble model"""
    print(f"\n🔍 Testing Ensemble Model")
    print("-" * 50)
    
    try:
        # Add ensemble module path
        ensemble_path = os.path.join(project_root, 'models', '04_ensemble')
        if ensemble_path not in sys.path:
            sys.path.insert(0, ensemble_path)
        
        # Import with absolute path handling
        import importlib.util
        ensemble_file = os.path.join(ensemble_path, 'ensemble_predictor.py')
        spec = importlib.util.spec_from_file_location("ensemble_predictor", ensemble_file)
        ensemble_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ensemble_module)
        EnsemblePredictor = ensemble_module.EnsemblePredictor
        
        model_paths = [
            os.path.join(project_root, 'models', '03_specialized_sssalud', 'model.pth'),
            os.path.join(project_root, 'models', '02_advanced', 'model.pth')
        ]
        weights = [0.7, 0.3]
        
        ensemble = EnsemblePredictor(model_paths, weights)
        
        test_images = [
            (os.path.join(project_root, "my_captchas", "1.png"), "UKhGh9"),
            (os.path.join(project_root, "my_captchas", "4.png"), "cGfFE2"),
            (os.path.join(project_root, "my_captchas", "7.png"), "DUzp49"),
            (os.path.join(project_root, "my_captchas", "11.png"), "CzSLcN"),
            (os.path.join(project_root, "my_captchas", "20.png"), "Z3TUnp"),
        ]
        
        for img_path, real_value in test_images:
            if os.path.exists(img_path):
                prediction, confidence = ensemble.predict_ensemble(img_path)
                match = "✅" if prediction == real_value else "❌"
                print(f"{os.path.basename(img_path):12} | {prediction:8} | {real_value:8} | {confidence:.3f} | {match}")
                
    except Exception as e:
        print(f"❌ Error testing ensemble: {e}")

def test_ensemble_with_postprocessing():
    """Test ensemble with post-processing"""
    print(f"\n🔍 Testing Ensemble + Post-processing")
    print("-" * 50)
    
    try:
        # Import ensemble predictor
        ensemble_path = os.path.join(project_root, 'models', '04_ensemble')
        ensemble_file = os.path.join(ensemble_path, 'ensemble_predictor.py')
        spec_ensemble = importlib.util.spec_from_file_location("ensemble_predictor", ensemble_file)
        ensemble_module = importlib.util.module_from_spec(spec_ensemble)
        spec_ensemble.loader.exec_module(ensemble_module)
        EnsemblePredictor = ensemble_module.EnsemblePredictor
        
        # Import postprocessor
        postprocessing_path = os.path.join(project_root, 'experiments', 'postprocessing')
        postprocessor_file = os.path.join(postprocessing_path, 'captcha_postprocessor.py')
        spec_post = importlib.util.spec_from_file_location("captcha_postprocessor", postprocessor_file)
        post_module = importlib.util.module_from_spec(spec_post)
        spec_post.loader.exec_module(post_module)
        CaptchaPostProcessor = post_module.CaptchaPostProcessor
        
        model_paths = [
            os.path.join(project_root, 'models', '03_specialized_sssalud', 'model.pth'),
            os.path.join(project_root, 'models', '02_advanced', 'model.pth')
        ]
        weights = [0.7, 0.3]
        
        ensemble = EnsemblePredictor(model_paths, weights)
        postprocessor = CaptchaPostProcessor()
        
        test_images = [
            (os.path.join(project_root, "my_captchas", "1.png"), "UKhGh9"),
            (os.path.join(project_root, "my_captchas", "4.png"), "cGfFE2"),
            (os.path.join(project_root, "my_captchas", "7.png"), "DUzp49"),
            (os.path.join(project_root, "my_captchas", "11.png"), "CzSLcN"),
            (os.path.join(project_root, "my_captchas", "20.png"), "Z3TUnp"),
        ]
        
        for img_path, real_value in test_images:
            if os.path.exists(img_path):
                raw_prediction, confidence = ensemble.predict_ensemble(img_path)
                corrected = postprocessor.apply_corrections(raw_prediction)
                
                raw_match = "✅" if raw_prediction == real_value else "❌"
                corrected_match = "✅" if corrected == real_value else "❌"
                
                print(f"{os.path.basename(img_path):12} | Raw: {raw_prediction:8} {raw_match} | Corrected: {corrected:8} {corrected_match}")
                
    except Exception as e:
        print(f"❌ Error testing ensemble with post-processing: {e}")

def main():
    """Main testing function"""
    print("🚀 PyCAPTCHA Model Testing Suite")
    print("=" * 60)
    
    # Test images (first 5 for quick testing)
    test_images = [
        (os.path.join(project_root, "my_captchas", "1.png"), "UKhGh9"),
        (os.path.join(project_root, "my_captchas", "4.png"), "cGfFE2"), 
        (os.path.join(project_root, "my_captchas", "7.png"), "DUzp49"),
        (os.path.join(project_root, "my_captchas", "11.png"), "CzSLcN"),
        (os.path.join(project_root, "my_captchas", "20.png"), "Z3TUnp"),
    ]
    
    # Test individual models
    models = [
        ("Baseline v1", os.path.join(project_root, "models", "01_baseline", "model.pth")),
        ("Advanced v2", os.path.join(project_root, "models", "02_advanced", "model.pth")),
        ("Specialized v3 ⭐", os.path.join(project_root, "models", "03_specialized_sssalud", "model.pth")),
    ]
    
    for model_name, model_path in models:
        test_model(model_name, model_path, test_images)
    
    # Test ensemble
    test_ensemble()
    
    # Test ensemble with post-processing
    test_ensemble_with_postprocessing()
    
    print(f"\n✅ Testing complete!")

if __name__ == "__main__":
    main()
