#!/usr/bin/env python3
"""
Ensemble predictor combining multiple models for better accuracy
"""
import torch
import numpy as np
from collections import Counter
from PIL import Image
import torchvision.transforms as transforms
from model.model import captcha_model, model_resnet
from data.dataset import CHAR_LEN, CLASS_NUM, str_to_lst, lst_to_str

class EnsemblePredictor:
    def __init__(self, model_paths, weights=None):
        """
        Initialize ensemble with multiple model paths
        weights: list of weights for each model (optional)
        """
        self.models = []
        self.weights = weights or [1.0] * len(model_paths)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((60, 160)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load models
        for path in model_paths:
            model = self._load_model(path)
            self.models.append(model)
    
    def _load_model(self, model_path):
        """Load a single model from checkpoint"""
        try:
            model = captcha_model(model_resnet())
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self.device)
            return model
        except:
            return None
    
    def predict_single(self, image_path, model):
        """Predict using a single model"""
        if model is None:
            return None, 0.0
            
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                # outputs shape: (1, CHAR_LEN, CLASS_NUM)
                probabilities = torch.softmax(outputs, dim=-1)
                predictions = torch.argmax(outputs, dim=-1)
                
                # Convert to string
                pred_str = lst_to_str(predictions[0])
                
                # Calculate confidence (average max probability per position)
                confidence = torch.max(probabilities, dim=-1)[0].mean().item()
                
                return pred_str, confidence
                
        except:
            return None, 0.0
    
    def predict_ensemble(self, image_path, method='weighted_voting'):
        """
        Predict using ensemble method
        method: 'voting', 'weighted_voting', 'confidence'
        """
        predictions = []
        confidences = []
        
        # Get predictions from all models
        for i, model in enumerate(self.models):
            pred, conf = self.predict_single(image_path, model)
            if pred is not None:
                predictions.append(pred)
                confidences.append(conf)
        
        if not predictions:
            return "ERROR!", 0.0
        
        if method == 'voting':
            return self._majority_voting(predictions), np.mean(confidences)
        elif method == 'weighted_voting':
            return self._weighted_voting(predictions, self.weights[:len(predictions)]), np.mean(confidences)
        elif method == 'confidence':
            return self._confidence_based(predictions, confidences), max(confidences)
    
    def _majority_voting(self, predictions):
        """Simple majority voting for each character position"""
        result = ""
        for pos in range(6):  # 6 characters
            chars = [pred[pos] for pred in predictions if len(pred) > pos]
            if chars:
                most_common = Counter(chars).most_common(1)[0][0]
                result += most_common
        return result
    
    def _weighted_voting(self, predictions, weights):
        """Weighted voting based on model weights"""
        result = ""
        for pos in range(6):
            char_weights = {}
            for pred, weight in zip(predictions, weights):
                if len(pred) > pos:
                    char = pred[pos]
                    char_weights[char] = char_weights.get(char, 0) + weight
            if char_weights:
                best_char = max(char_weights, key=char_weights.get)
                result += best_char
        return result
    
    def _confidence_based(self, predictions, confidences):
        """Choose prediction from most confident model"""
        if confidences:
            best_idx = np.argmax(confidences)
            return predictions[best_idx]
        return predictions[0] if predictions else ""

# Example usage and testing
if __name__ == "__main__":
    import os
    
    model_paths = []
    if os.path.exists("../03_specialized_sssalud/model.pth"):
        model_paths.append("../03_specialized_sssalud/model.pth")
    if os.path.exists("../02_advanced/model.pth"):
        model_paths.append("../02_advanced/model.pth")
    
    if not model_paths:
        print("No models found")
        exit()
    
    weights = [0.7, 0.3] if len(model_paths) == 2 else [1.0]
    ensemble = EnsemblePredictor(model_paths, weights)
    
    test_cases = [
        ("../../my_captchas/1.png", "UKhGh9"),
        ("../../my_captchas/4.png", "cGfFE2"),
        ("../../my_captchas/7.png", "DUzp49"),
        ("../../my_captchas/11.png", "CzSLcN"),
        ("../../my_captchas/20.png", "Z3TUnp"),
    ]
    
    for image_path, real_value in test_cases:
        if os.path.exists(image_path):
            # Individual predictions
            for i, model in enumerate(ensemble.models):
                pred, conf = ensemble.predict_single(image_path, model)
                model_name = "Specialized" if i == 0 else "Advanced"
                print(f"{model_name}: {pred}")
            
            # Ensemble prediction
            ensemble_pred, _ = ensemble.predict_ensemble(image_path, method='weighted_voting')
            print(f"Ensemble: {ensemble_pred}")
            print(f"Real: {real_value}")
            print("---")
