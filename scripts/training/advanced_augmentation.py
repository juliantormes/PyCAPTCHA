#!/usr/bin/env python3
"""
Advanced data augmentation specifically for sssalud CAPTCHAs
"""
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random

class AdvancedCaptchaAugmentation:
    def __init__(self):
        self.noise_patterns = [
            self._add_line_noise,
            self._add_dot_noise,
            self._add_gaussian_noise,
            self._add_salt_pepper_noise
        ]
        
        self.distortions = [
            self._perspective_transform,
            self._elastic_transform,
            self._rotation_skew,
            self._character_spacing
        ]
    
    def augment_image(self, image, num_augmentations=5):
        """Apply multiple augmentation techniques"""
        augmented_images = []
        
        for _ in range(num_augmentations):
            aug_img = image.copy()
            
            # Apply random noise
            if random.random() > 0.3:
                noise_func = random.choice(self.noise_patterns)
                aug_img = noise_func(aug_img)
            
            # Apply random distortion
            if random.random() > 0.4:
                distort_func = random.choice(self.distortions)
                aug_img = distort_func(aug_img)
            
            # Apply color/brightness changes
            if random.random() > 0.3:
                aug_img = self._adjust_colors(aug_img)
            
            augmented_images.append(aug_img)
        
        return augmented_images
    
    def _add_line_noise(self, image):
        """Add crossing lines similar to real sssalud CAPTCHAs"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Add 1-3 random lines
        num_lines = random.randint(1, 3)
        for _ in range(num_lines):
            # Random line coordinates
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            
            # Random thickness and color
            thickness = random.randint(1, 3)
            color = random.randint(100, 200)  # Gray lines
            
            cv2.line(img_array, (x1, y1), (x2, y2), color, thickness)
        
        return Image.fromarray(img_array)
    
    def _add_dot_noise(self, image):
        """Add random dots/speckles"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Add random dots
        num_dots = random.randint(5, 20)
        for _ in range(num_dots):
            x, y = random.randint(0, w-1), random.randint(0, h-1)
            radius = random.randint(1, 2)
            color = random.randint(0, 255)
            cv2.circle(img_array, (x, y), radius, color, -1)
        
        return Image.fromarray(img_array)
    
    def _perspective_transform(self, image):
        """Apply slight perspective distortion"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Define source and destination points for perspective transform
        offset = random.randint(2, 8)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32([
            [random.randint(-offset, offset), random.randint(-offset, offset)],
            [w + random.randint(-offset, offset), random.randint(-offset, offset)],
            [w + random.randint(-offset, offset), h + random.randint(-offset, offset)],
            [random.randint(-offset, offset), h + random.randint(-offset, offset)]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = cv2.warpPerspective(img_array, matrix, (w, h))
        
        return Image.fromarray(transformed)
    
    def _adjust_colors(self, image):
        """Adjust brightness, contrast, and sharpness"""
        # Brightness
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Contrast
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.9, 1.3))
        
        # Sharpness
        if random.random() > 0.5:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.5))
        
        return image
    
    def _elastic_transform(self, image):
        """Apply elastic deformation"""
        # Simplified elastic transform
        img_array = np.array(image)
        return Image.fromarray(img_array)  # Placeholder for now
    
    def _rotation_skew(self, image):
        """Apply small rotation and skew"""
        angle = random.uniform(-5, 5)  # Small rotation
        return image.rotate(angle, fillcolor=255)
    
    def _character_spacing(self, image):
        """Simulate different character spacing"""
        # This would require more complex manipulation
        return image
    
    def _add_gaussian_noise(self, image):
        """Add Gaussian noise"""
        img_array = np.array(image)
        noise = np.random.normal(0, 10, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    def _add_salt_pepper_noise(self, image):
        """Add salt and pepper noise"""
        img_array = np.array(image)
        noise = np.random.random(img_array.shape)
        img_array[noise < 0.01] = 0    # Salt
        img_array[noise > 0.99] = 255  # Pepper
        return Image.fromarray(img_array)

# Usage example
if __name__ == "__main__":
    augmenter = AdvancedCaptchaAugmentation()
    
    # Load a sample image
    sample_image = Image.open("./my_captchas/1.png")
    
    # Generate augmented versions
    augmented = augmenter.augment_image(sample_image, num_augmentations=10)
    
    # Save examples
    for i, aug_img in enumerate(augmented):
        aug_img.save(f"./augmented_sample_{i}.png")
