from captcha.image import ImageCaptcha
import random
import os
from tqdm import trange
import sys
sys.path.append('utils')
from config_util import configGetter

cfg = configGetter('DATASET')

# Create ImageCaptcha with different styling to match sssalud
image = ImageCaptcha(
    fonts=[cfg['CAPTCHA']['FONT_DIR']], 
    width=240, 
    height=60
)

def sssaludStyleSeqGenerator(captcha_len):
    """Generate CAPTCHA sequences similar to sssalud style"""
    ret = ""
    for i in range(captcha_len):
        # Based on your examples, sssalud uses:
        # - Uppercase letters (more common)
        # - Lowercase letters 
        # - Numbers
        # Let's adjust the probability to match real patterns
        
        num = chr(random.randint(48,57))        # 0-9
        lower = chr(random.randint(97, 122))    # a-z  
        upper = chr(random.randint(65, 90))     # A-Z
        
        # Weighted selection based on observed patterns
        choices = [upper] * 4 + [lower] * 3 + [num] * 3  # 40% upper, 30% lower, 30% numbers
        s = str(random.choice(choices))
        ret += s
    return ret

def captchaGenerator(dataset_path, dataset_len, captcha_len):
    os.makedirs(dataset_path, exist_ok=True)
    for i in trange(dataset_len):
        char_seq = sssaludStyleSeqGenerator(captcha_len)
        save_path = os.path.join(dataset_path, f'{char_seq}.{i}.png')
        image.write(char_seq, save_path)

def generateSssaludStyleCaptcha():
    """Generate training data that better matches sssalud CAPTCHA style"""
    TRAINING_DIR = './dataset_sssalud/train'
    TESTING_DIR = './dataset_sssalud/val'
    TRAINING_DATASET_LEN = 50000  # Start with smaller dataset for testing
    TESTING_DATASET_LEN = 1000
    CHAR_LEN = 6

    print("🎯 Generating sssalud-style CAPTCHA dataset...")
    print(f"Training samples: {TRAINING_DATASET_LEN}")
    print(f"Validation samples: {TESTING_DATASET_LEN}")
    
    captchaGenerator(TRAINING_DIR, TRAINING_DATASET_LEN, CHAR_LEN)
    captchaGenerator(TESTING_DIR, TESTING_DATASET_LEN, CHAR_LEN)
    print("✅ sssalud-style dataset generation complete!")
    
if __name__ == "__main__":
    generateSssaludStyleCaptcha()
