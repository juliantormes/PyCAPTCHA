#!/usr/bin/env python3
"""
Custom loss function for character-level optimization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharacterFocusedLoss(nn.Module):
    def __init__(self, position_weights=None, char_weights=None):
        super().__init__()
        self.position_weights = position_weights or [1.0] * 6
        self.char_weights = char_weights or {}
        self.base_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        predictions: (batch_size, num_chars, vocab_size)
        targets: (batch_size, num_chars)
        """
        batch_size, num_chars, vocab_size = predictions.shape
        total_loss = 0
        
        for pos in range(num_chars):
            # Calculate loss for this position
            pos_pred = predictions[:, pos, :]  # (batch_size, vocab_size)
            pos_target = targets[:, pos]       # (batch_size,)
            
            pos_loss = self.base_loss(pos_pred, pos_target)
            
            # Apply position weight
            pos_loss = pos_loss * self.position_weights[pos]
            
            total_loss += pos_loss.mean()
        
        return total_loss / num_chars

class FocalLoss(nn.Module):
    """Focal loss to handle hard examples better"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """Combine multiple loss functions"""
    def __init__(self):
        super().__init__()
        self.char_loss = CharacterFocusedLoss(
            position_weights=[1.2, 0.8, 1.0, 1.1, 0.9, 1.0]  # Based on your analysis
        )
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.sequence_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        # Reshape for different loss calculations
        batch_size, num_chars, vocab_size = predictions.shape
        
        # Character-level loss
        char_loss = self.char_loss(predictions, targets)
        
        # Focal loss (flatten first)
        pred_flat = predictions.view(-1, vocab_size)
        target_flat = targets.view(-1)
        focal_loss = self.focal_loss(pred_flat, target_flat)
        
        # Combine losses
        total_loss = 0.6 * char_loss + 0.4 * focal_loss
        
        return total_loss
