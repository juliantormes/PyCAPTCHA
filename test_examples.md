# 🧪 Testing Your Own CAPTCHA Images

## Quick Test Commands

### Test with sample images (to verify everything works):
```bash
# Activate environment and test sample 1
.\.venv\Scripts\activate && python predictor.py --input ./assets/captcha.png --ckpt ./checkpoint/model.pth

# Test sample 2
.\.venv\Scripts\activate && python predictor.py --input ./assets/testing.png --ckpt ./checkpoint/model.pth
```

### Test with your own images:
```bash
# Example 1: Image on Desktop
.\.venv\Scripts\activate && python predictor.py --input "C:\Users\Julian\Desktop\my_captcha.png" --ckpt ./checkpoint/model.pth

# Example 2: Image in Downloads
.\.venv\Scripts\activate && python predictor.py --input "C:\Users\Julian\Downloads\captcha_test.jpg" --ckpt ./checkpoint/model.pth

# Example 3: Multiple images in a folder
.\.venv\Scripts\activate && python predictor.py --input "./my_captchas/test1.png" --ckpt ./checkpoint/model.pth
.\.venv\Scripts\activate && python predictor.py --input "./my_captchas/test2.png" --ckpt ./checkpoint/model.pth
```

## Expected Output Format:
```
Predicted CAPTCHA: abc123
```

## Troubleshooting:

### If you get "file not found" error:
- Check the file path is correct
- Use quotes around paths with spaces: `"C:\Users\Julian\My Folder\image.png"`
- Use forward slashes or double backslashes: `C:\\Users\\Julian\\image.png`

### If predictions seem wrong:
- Make sure the CAPTCHA contains exactly 6 characters
- Ensure good image quality (not too blurry)
- The model works best with similar style to training data

### For batch testing multiple images:
Create a batch script or use a loop to test multiple images at once.

## Image Requirements:
- ✅ **Format:** PNG, JPG, JPEG, BMP, GIF
- ✅ **Content:** 6-character CAPTCHAs (letters a-z, numbers 0-9)
- ✅ **Size:** Any size (automatically resized to 160x60)
- ✅ **Quality:** Clear, readable text
