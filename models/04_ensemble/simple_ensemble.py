#!/usr/bin/env python3
"""
Simple ensemble using the existing predictor.py
"""
import subprocess
import os
from collections import Counter

class SimpleEnsemble:
    def __init__(self, model_paths, weights=None):
        self.model_paths = model_paths
        self.weights = weights or [1.0] * len(model_paths)
    
    def predict_with_model(self, image_path, model_path):
        """Run predictor.py with specific model"""
        try:
            cmd = f"python predictor.py --input {image_path} --ckpt {model_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Extract prediction from output
                output = result.stdout.strip()
                lines = output.split('\n')
                for line in lines:
                    if 'Prediction:' in line:
                        pred = line.split('Prediction:')[-1].strip()
                        return pred
                    # Sometimes it just outputs the prediction directly
                    if len(line.strip()) == 6 and line.strip().isalnum():
                        return line.strip()
            
            return None
        except Exception as e:
            print(f"Error with model {model_path}: {e}")
            return None
    
    def predict_ensemble(self, image_path):
        """Get ensemble prediction"""
        predictions = []
        
        for model_path in self.model_paths:
            pred = self.predict_with_model(image_path, model_path)
            if pred and len(pred) == 6:
                predictions.append(pred)
        
        if not predictions:
            return "ERROR!"
        
        # If only one prediction, return it
        if len(predictions) == 1:
            return predictions[0]
        
        # Weighted voting
        result = ""
        for pos in range(6):
            char_weights = {}
            for pred, weight in zip(predictions, self.weights[:len(predictions)]):
                char = pred[pos]
                char_weights[char] = char_weights.get(char, 0) + weight
            
            best_char = max(char_weights, key=char_weights.get)
            result += best_char
        
        return result

def test_ensemble():
    """Test the ensemble on some samples"""
    model_paths = []
    
    if os.path.exists("./checkpoints_sssalud/model.pth"):
        model_paths.append("./checkpoints_sssalud/model.pth")
    if os.path.exists("./checkpoints_advanced/model.pth"):
        model_paths.append("./checkpoints_advanced/model.pth")
    
    if not model_paths:
        print("❌ No models found!")
        return
    
    # Give more weight to specialized model
    weights = [0.7, 0.3] if len(model_paths) == 2 else [1.0]
    
    ensemble = SimpleEnsemble(model_paths, weights)
    
    print(f"🤖 Testing Simple Ensemble with {len(model_paths)} models")
    print("=" * 60)
    
    # Test cases with 20 samples
    test_cases = [
        ("./my_captchas/1.png", "UKhGh9"),
        ("./my_captchas/4.png", "cGfFE2"),
        ("./my_captchas/7.png", "DUzp49"),
        ("./my_captchas/11.png", "CzSLcN"),
        ("./my_captchas/15.png", "z56VDI"),
        ("./my_captchas/20.png", "Z3TUnp"),
    ]
    
    correct = 0
    total = 0
    
    for image_path, real_value in test_cases:
        if os.path.exists(image_path):
            print(f"\n🔍 Testing {image_path} (Real: {real_value})")
            
            # Individual model predictions
            for i, model_path in enumerate(model_paths):
                pred = ensemble.predict_with_model(image_path, model_path)
                model_name = "Specialized" if "sssalud" in model_path else "Advanced"
                match = "✅" if pred == real_value else "❌"
                print(f"  {model_name}: {pred} {match}")
            
            # Ensemble prediction
            ensemble_pred = ensemble.predict_ensemble(image_path)
            ensemble_match = "✅" if ensemble_pred == real_value else "❌"
            print(f"  🤖 Ensemble: {ensemble_pred} {ensemble_match}")
            
            total += 1
            if ensemble_pred == real_value:
                correct += 1
    
    if total > 0:
        accuracy = correct / total * 100
        print(f"\n📊 Ensemble Results: {correct}/{total} correct ({accuracy:.1f}%)")

if __name__ == "__main__":
    test_ensemble()
