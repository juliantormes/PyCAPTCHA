#!/usr/bin/env python3
"""
Generador de CAPTCHA sintético avanzado
Basado en análisis de 20 CAPTCHAs reales de sssalud
Genera datos con vocabulario completo: 0-9, a-z, A-Z (62 clases)
"""

import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from typing import List, Tuple
import cv2

class AdvancedCaptchaGenerator:
    def __init__(self, width: int = 224, height: int = 64):
        self.width = width
        self.height = height
        
        # Vocabulario completo - 62 caracteres
        self.charset = string.digits + string.ascii_lowercase + string.ascii_uppercase
        print(f"📚 Vocabulario: {self.charset}")
        print(f"📏 Total caracteres: {len(self.charset)}")
        
        # Configuración basada en análisis de CAPTCHAs reales
        self.captcha_length = 6
        
        # Colores observados en CAPTCHAs reales
        self.text_colors = [
            (0, 0, 0),       # Negro
            (50, 50, 50),    # Gris oscuro  
            (80, 80, 80),    # Gris medio
        ]
        
        self.bg_colors = [
            (240, 240, 240), # Gris muy claro
            (250, 250, 250), # Casi blanco
            (245, 245, 245), # Blanco grisáceo
        ]
        
        # Configuración de distorsiones
        self.noise_intensity = 0.3
        self.distortion_strength = 0.2
        
        # Cargar fuentes (usar fuentes del sistema)
        self.fonts = self._load_fonts()
        
    def _load_fonts(self) -> List:
        """Cargar fuentes disponibles del sistema"""
        fonts = []
        
        # Fuentes comunes en Windows
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf", 
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/verdana.ttf",
            "fonts/font1.ttf"  # Fuente del proyecto
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    # Diferentes tamaños para variación
                    for size in [28, 32, 36, 40]:
                        fonts.append(ImageFont.truetype(font_path, size))
                except:
                    continue
        
        # Fuente por defecto si no se encuentra ninguna
        if not fonts:
            fonts = [ImageFont.load_default() for _ in range(4)]
            
        print(f"🔤 Fuentes cargadas: {len(fonts)}")
        return fonts
    
    def generate_text(self) -> str:
        """Generar texto aleatorio con distribución balanceada"""
        text = ""
        for _ in range(self.captcha_length):
            text += random.choice(self.charset)
        return text
    
    def add_noise(self, image: Image.Image) -> Image.Image:
        """Añadir ruido realista al CAPTCHA"""
        img_array = np.array(image)
        
        # Ruido gaussiano
        noise = np.random.normal(0, self.noise_intensity * 255, img_array.shape)
        noisy_img = img_array.astype(np.float32) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        # Líneas de interferencia
        img_pil = Image.fromarray(noisy_img)
        draw = ImageDraw.Draw(img_pil)
        
        # 2-4 líneas aleatorias
        num_lines = random.randint(2, 4)
        for _ in range(num_lines):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = random.randint(0, self.width)
            y2 = random.randint(0, self.height)
            
            color = random.choice([(100, 100, 100), (150, 150, 150), (200, 200, 200)])
            width = random.randint(1, 2)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        
        # Puntos aleatorios
        num_dots = random.randint(50, 100)
        for _ in range(num_dots):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            color = random.choice([(120, 120, 120), (180, 180, 180)])
            draw.point((x, y), fill=color)
        
        return img_pil
    
    def apply_distortion(self, image: Image.Image) -> Image.Image:
        """Aplicar distorsiones geométricas"""
        img_array = np.array(image)
        
        # Distorsión ondulada horizontal
        rows, cols = img_array.shape[:2]
        
        # Crear mapas de distorsión
        map_x = np.zeros((rows, cols), dtype=np.float32)
        map_y = np.zeros((rows, cols), dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
                # Ondulación horizontal
                offset_x = self.distortion_strength * 10 * np.sin(2 * np.pi * i / 30)
                offset_y = self.distortion_strength * 5 * np.sin(2 * np.pi * j / 50)
                
                map_x[i, j] = j + offset_x
                map_y[i, j] = i + offset_y
        
        # Aplicar remapping
        distorted = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)
        
        return Image.fromarray(distorted)
    
    def generate_single_captcha(self, text: str = None) -> Tuple[Image.Image, str]:
        """Generar un CAPTCHA individual"""
        if text is None:
            text = self.generate_text()
        
        # Crear imagen base
        bg_color = random.choice(self.bg_colors)
        image = Image.new('RGB', (self.width, self.height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Configuración de texto
        text_color = random.choice(self.text_colors)
        font = random.choice(self.fonts)
        
        # Calcular posicionamiento dinámico
        char_width = self.width // (self.captcha_length + 1)
        start_x = char_width // 2
        
        # Dibujar cada carácter con variaciones
        for i, char in enumerate(text):
            # Posición con variación aleatoria
            x = start_x + i * char_width + random.randint(-8, 8)
            y = self.height // 2 - 15 + random.randint(-5, 5)
            
            # Rotación ligera
            rotation = random.randint(-15, 15)
            
            # Crear imagen temporal para el carácter rotado
            char_img = Image.new('RGBA', (50, 50), (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_img)
            char_draw.text((25, 25), char, font=font, fill=text_color, anchor="mm")
            
            # Rotar carácter
            if rotation != 0:
                char_img = char_img.rotate(rotation, expand=True)
            
            # Pegar en imagen principal
            image.paste(char_img, (x-25, y-25), char_img)
        
        # Aplicar efectos
        if random.random() < 0.7:  # 70% probabilidad de ruido
            image = self.add_noise(image)
        
        if random.random() < 0.5:  # 50% probabilidad de distorsión
            image = self.apply_distortion(image)
        
        # Filtros adicionales
        if random.random() < 0.3:  # 30% blur ligero
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return image, text
    
    def generate_dataset(self, num_samples: int, output_dir: str):
        """Generar dataset completo"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎯 Generando {num_samples} CAPTCHAs sintéticos...")
        print(f"📁 Directorio: {output_dir}")
        
        # Asegurar distribución balanceada de caracteres
        samples_per_char = max(1, num_samples // len(self.charset))
        
        generated = 0
        
        # Generar muestras balanceadas
        for char in self.charset:
            for i in range(samples_per_char):
                # Crear texto que incluya el carácter específico
                text = self.generate_text()
                # Asegurar que el carácter aparezca al menos una vez
                pos = random.randint(0, self.captcha_length - 1)
                text = text[:pos] + char + text[pos+1:]
                
                image, label = self.generate_single_captcha(text)
                
                # Guardar imagen
                filename = f"{label}_{generated:06d}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                
                generated += 1
                
                if generated % 1000 == 0:
                    print(f"  ✅ Generados: {generated}")
                
                if generated >= num_samples:
                    break
            
            if generated >= num_samples:
                break
        
        # Llenar hasta el número objetivo con muestras aleatorias
        while generated < num_samples:
            image, label = self.generate_single_captcha()
            
            filename = f"{label}_{generated:06d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            
            generated += 1
            
            if generated % 1000 == 0:
                print(f"  ✅ Generados: {generated}")
        
        print(f"🎉 Dataset completo: {generated} muestras")
        
        # Estadísticas del dataset
        self._print_dataset_stats(output_dir)
    
    def _print_dataset_stats(self, dataset_dir: str):
        """Imprimir estadísticas del dataset generado"""
        files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
        
        char_counts = {char: 0 for char in self.charset}
        
        for filename in files:
            label = filename.split('_')[0]
            for char in label:
                if char in char_counts:
                    char_counts[char] += 1
        
        print(f"\n📊 ESTADÍSTICAS DEL DATASET:")
        print(f"Total imágenes: {len(files)}")
        
        # Mostrar distribución por tipo
        digits = sum(char_counts[c] for c in string.digits)
        lowercase = sum(char_counts[c] for c in string.ascii_lowercase)
        uppercase = sum(char_counts[c] for c in string.ascii_uppercase)
        
        print(f"Dígitos (0-9): {digits}")
        print(f"Minúsculas (a-z): {lowercase}")
        print(f"Mayúsculas (A-Z): {uppercase}")
        print(f"Total caracteres: {digits + lowercase + uppercase}")

def main():
    """Función principal para generar dataset"""
    generator = AdvancedCaptchaGenerator()
    
    # Configuración del dataset
    train_samples = 50000  # 50k para entrenamiento
    val_samples = 5000     # 5k para validación
    
    # Crear directorios
    base_dir = "dataset_synthetic_v2"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    # Generar datasets
    print("🚀 GENERACIÓN DE DATASET SINTÉTICO AVANZADO")
    print("=" * 50)
    
    generator.generate_dataset(train_samples, train_dir)
    print("\n" + "="*30)
    generator.generate_dataset(val_samples, val_dir)
    
    print(f"\n🎉 DATASET COMPLETO GENERADO EN: {base_dir}")
    print(f"📈 Training: {train_samples} muestras")
    print(f"📊 Validation: {val_samples} muestras")
    print(f"📚 Vocabulario: 62 caracteres (0-9, a-z, A-Z)")

if __name__ == "__main__":
    main()
