#!/usr/bin/env python3
"""
Advanced sssalud-style CAPTCHA generator with improved realism
Focuses on visual features that match real sssalud CAPTCHAs
"""

import random
import string
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from tqdm import tqdm

def generate_advanced_sssalud_captcha():
    """Generate more realistic sssalud-style CAPTCHA"""
    
    # Character distribution based on real analysis
    position_weights = {
        0: {'upper': 0.30, 'lower': 0.50, 'digit': 0.20},
        1: {'upper': 0.40, 'lower': 0.20, 'digit': 0.40},
        2: {'upper': 0.20, 'lower': 0.60, 'digit': 0.20},
        3: {'upper': 0.50, 'lower': 0.30, 'digit': 0.20},
        4: {'upper': 0.30, 'lower': 0.40, 'digit': 0.30},
        5: {'upper': 0.40, 'lower': 0.40, 'digit': 0.20}
    }
    
    chars = []
    for pos in range(6):
        rand = random.random()
        weights = position_weights[pos]
        
        if rand < weights['upper']:
            chars.append(random.choice(string.ascii_uppercase))
        elif rand < weights['upper'] + weights['lower']:
            chars.append(random.choice(string.ascii_lowercase))
        else:
            chars.append(random.choice(string.digits))
    
    return ''.join(chars)

def create_advanced_training_image(text, save_path):
    """Create a more realistic CAPTCHA image matching sssalud style"""
    
    # Image dimensions
    width, height = 240, 60
    
    # Create image with white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Font path
    font_path = './fonts/font1.ttf'
    
    try:
        # Vary font size slightly for realism
        font_size = random.randint(32, 38)
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    
    # Character positioning with slight randomness
    base_x = 10
    char_spacing = 35
    base_y = random.randint(8, 15)
    
    # Add some background noise (light)
    for _ in range(random.randint(50, 100)):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        color = random.randint(200, 240)
        draw.point((x, y), fill=(color, color, color))
    
    # Draw each character with slight variations
    for i, char in enumerate(text):
        # Position with small random offset
        x = base_x + i * char_spacing + random.randint(-3, 3)
        y = base_y + random.randint(-3, 3)
        
        # Character color - mostly black with slight gray variations
        color_val = random.randint(0, 30)
        char_color = (color_val, color_val, color_val)
        
        draw.text((x, y), char, font=font, fill=char_color)
    
    # Apply slight distortions to make it more realistic
    
    # 1. Add very light blur occasionally
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # 2. Add subtle rotation
    if random.random() < 0.4:
        angle = random.uniform(-2, 2)
        img = img.rotate(angle, fillcolor='white')
    
    # 3. Add slight noise lines (very subtle)
    if random.random() < 0.5:
        for _ in range(random.randint(1, 3)):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            line_color = random.randint(180, 220)
            draw.line([(x1, y1), (x2, y2)], fill=(line_color, line_color, line_color), width=1)
    
    # Save the image
    img.save(save_path)

def generate_advanced_dataset():
    """Generate an advanced training dataset"""
    
    print("🚀 Generating ADVANCED sssalud-style CAPTCHA dataset...")
    
    # Create directories
    dataset_dir = './dataset_sssalud_v2'
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Generate training data (increased amount)
    print("Generating training samples...")
    for i in tqdm(range(100000), desc="Training"):  # Increased to 100k
        captcha_text = generate_advanced_sssalud_captcha()
        filename = f"{captcha_text}.png"
        filepath = os.path.join(train_dir, filename)
        create_advanced_training_image(captcha_text, filepath)
    
    # Generate validation data
    print("Generating validation samples...")
    for i in tqdm(range(2000), desc="Validation"):
        captcha_text = generate_advanced_sssalud_captcha()
        filename = f"{captcha_text}.png"
        filepath = os.path.join(val_dir, filename)
        create_advanced_training_image(captcha_text, filepath)
    
    print("✅ Advanced dataset generation complete!")
    print(f"Training samples: 100,000")
    print(f"Validation samples: 2,000")
    print(f"Dataset saved to: {dataset_dir}")

if __name__ == "__main__":
    generate_advanced_dataset()
