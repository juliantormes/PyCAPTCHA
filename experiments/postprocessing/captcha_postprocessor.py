#!/usr/bin/env python3
"""
Post-processing rules to improve CAPTCHA predictions based on 20 sample analysis
"""
import re

class CaptchaPostProcessor:
    def __init__(self):
        # Specific corrections for the weird outputs we're seeing
        self.direct_corrections = {
            'z': '2',  # z is often confused with 2
            'e': '3',  # e is often confused with 3
            'o': '0',  # o is often confused with 0
            'n': 'N',  # n should be uppercase N
            'w': 'W',  # w should be uppercase W
            'b': 'B',  # b should be uppercase B
        }
        
        # Valid characters from 20 samples
        self.valid_chars = set(['2', '3', '4', '5', '6', '7', '8', '9', 
                               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 
                               'I', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 
                               'T', 'U', 'V', 'W', 'Y', 'Z', 
                               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
                               'k', 'm', 'n', 'p', 'r', 't', 'u', 'v', 
                               'w', 'y', 'z'])
        
        # Most common characters by position (from analysis)
        self.position_common = {
            0: ['U', 'C', 'D', 'M', 'T', 'Z', 'c', 'e', 'g', 'p', 'r', 't', 'v', 'w', 'z', '2', '3', '9'],
            1: ['K', 'N', 'U', 'W', 'Z', 'h', 'm', 'n', 'p', 'v', 'z', '2', '3', '4', '5', '6', '7'],
            2: ['A', 'H', 'K', 'L', 'R', 'S', 'T', 'W', 'f', 'h', 'm', 'y', 'z', '2', '4', '6'],
            3: ['E', 'F', 'G', 'L', 'U', 'V', 'Y', 'a', 'd', 'f', 'k', 'p', 'v', '3', '7', '8'],
            4: ['B', 'D', 'E', 'H', 'U', 'Z', 'b', 'c', 'h', 'm', 'n', 'p', 'v', 'z', '2', '4'],
            5: ['C', 'D', 'E', 'F', 'I', 'N', 'P', 'S', 'e', 'm', 'p', 'r', 'u', 'w', '2', '3', '5', '9']
        }
    
    def apply_corrections(self, prediction):
        """Apply post-processing corrections"""
        if not prediction or len(prediction) != 6:
            return prediction
            
        corrected = list(prediction)
        
        # Check if prediction looks like garbage (repeated chars, invalid patterns)
        unique_chars = len(set(corrected))
        is_garbage = (
            unique_chars <= 2 or  # Too few unique characters
            all(c == 'z' or c == 'e' or c == 'o' for c in corrected[:5]) or  # Common garbage pattern
            corrected.count('e') >= 4 or  # Too many 'e's
            corrected.count('z') >= 3     # Too many 'z's
        )
        
        if is_garbage:
            # For garbage predictions, use position-based replacement
            for i in range(6):
                corrected[i] = self.position_common[i][0] if self.position_common[i] else 'A'
        else:
            # For reasonable predictions, only fix obvious issues
            for i, char in enumerate(corrected):
                if char in self.direct_corrections:
                    corrected[i] = self.direct_corrections[char]
                elif char not in self.valid_chars:
                    # Replace invalid chars with most common for that position
                    corrected[i] = self.position_common[i][0] if self.position_common[i] else 'A'
        
        return ''.join(corrected)
    
    def _find_best_replacement(self, char, position):
        """Find best replacement for character based on position"""
        return self.position_common[position][0] if self.position_common[position] else 'A'

# Test with ensemble
def test_postprocessor():
    processor = CaptchaPostProcessor()
    
    # Test with the weird predictions from ensemble
    test_cases = [
        ("zzeeeo", "UKhGh9"),
        ("zeeeeo", "cGfFE2"), 
        ("zzeebo", "DUzp49"),
        ("eseeeo", "CzSLcN"),
        ("zeeeeo", "Z3TUnp"),
    ]
    
    print("Post-processing test:")
    for prediction, real in test_cases:
        corrected = processor.apply_corrections(prediction)
        print(f"{prediction} -> {corrected} (Real: {real})")

if __name__ == "__main__":
    test_postprocessor()
