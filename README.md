# PyCAPTCHA 🔍
![版本号](https://img.shields.io/badge/Version-Beta--0.0.1-blue)
![issues](https://img.shields.io/github/issues/ZiYang-xie/PyCAPTCHA?style=plastic)
![forks](https://img.shields.io/github/forks/ZiYang-xie/PyCAPTCHA)
![stars](https://img.shields.io/github/stars/ZiYang-xie/PyCAPTCHA)
![License](https://img.shields.io/github/license/ZiYang-xie/PyCAPTCHA)

![](./assets/captcha.png)
---

**An End-to-end Pytorch-Lightning implemented CAPTCHA OCR model.**  
Training 2 epoch under 100k images to get over 96% acc on Val dataset 🤩  
> with 200k or even more training set you may get >98% acc

![](./assets/testing.png)



## INSTALL ⚙️
### Step 0: Clone the Project
```shell
git clone https://github.com/ZiYang-xie/PyCAPTCHA
cd PyCAPTCHA
```

### Step 1: Create & Activate Virtual Environment
```shell
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
# source venv/bin/activate
```

### Step 2: Install PyTorch with CUDA Support (Recommended for GPU training)
```shell
# For GPU training (CUDA 12.1)
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html

# OR for CPU-only training
pip install torch torchvision torchaudio
```

### Step 3: Install Other Requirements
```shell
pip install pytorch-lightning>=2.0.0 tensorboard pillow pyyaml numpy matplotlib
```

### Step 4: Verify Installation
```python
# Run this to verify CUDA is available (for GPU users)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start 🚀

### Generate Training Data
```shell
python utils/captcha_generater.py
```

### Train the Model
```shell
# With GPU (Recommended - much faster!)
python launcher.py --exp_name my_experiment --gpus 1

# With CPU only (slower)
python launcher.py --exp_name my_experiment --gpus 0

# Default (CPU) - not recommended for training
python launcher.py
```

**Performance Comparison:**
- 🚀 **GPU Training:** ~5.6 iterations/second (RTX 3070)
- 🐌 **CPU Training:** ~0.2 iterations/second  
- ⚡ **Speed Improvement:** ~28x faster with GPU!

### Test the Model
```shell
# Test on validation dataset
python test.py --ckpt ./checkpoint/model.pth

# Test on your own CAPTCHA images
python predictor.py --input path/to/your/captcha.png --ckpt ./checkpoint/model.pth

# Test with sample images
python predictor.py --input ./assets/captcha.png --ckpt ./checkpoint/model.pth
python predictor.py --input ./assets/testing.png --ckpt ./checkpoint/model.pth
```

**Supported Image Formats:** PNG, JPG, JPEG, BMP, GIF  
**Requirements:** Images should contain 6-character CAPTCHAs (letters and numbers)

## Document 📃
> Checkout the PyCAPTCHA Usage in WIKI Page
  
Check the [Doc](https://github.com/ZiYang-xie/PyCAPTCHA/wiki)
