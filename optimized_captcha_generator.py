#!/usr/bin/env python3
"""
Generador CAPTCHA optimizado basado en análisis de imágenes reales
Parámetros exactos extraídos de los 20 CAPTCHAs reales
"""

import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from typing import List, Tuple
import cv2

class OptimizedCaptchaGenerator:
    def __init__(self):
        # Parámetros exactos de CAPTCHAs reales
        self.width = 230   # Dimensión real exacta
        self.height = 80   # Dimensión real exacta
        
        # Vocabulario completo - 62 caracteres
        self.charset = string.digits + string.ascii_lowercase + string.ascii_uppercase
        self.captcha_length = 6
        
        # Colores extraídos de análisis real
        self.bg_colors = [
            (255, 255, 255),  # Blanco puro (19/20 CAPTCHAs)
            (206, 206, 206),  # Gris claro (1/20 CAPTCHAs)
            (250, 250, 250),  # Variación blanco
        ]
        
        # Colores de texto (típicamente negro y grises)
        self.text_colors = [
            (0, 0, 0),        # Negro puro
            (40, 40, 40),     # Gris muy oscuro
            (60, 60, 60),     # Gris oscuro
        ]
        
        # Parámetros de ruido calibrados
        self.noise_intensity = 0.35  # Del análisis real
        self.avg_lines = 385         # Líneas de interferencia reales
        
        # Distribución real de caracteres
        self.char_distribution = {
            'digits': 0.233,      # 23.3%
            'lowercase': 0.400,   # 40.0%
            'uppercase': 0.367,   # 36.7%
        }
        
        # Cargar fuentes
        self.fonts = self._load_fonts()
        
        print(f"🎯 Generador optimizado inicializado")
        print(f"📐 Dimensiones: {self.width}x{self.height}")
        print(f"📊 Distribución: dígitos {self.char_distribution['digits']:.1%}, "
              f"minúsculas {self.char_distribution['lowercase']:.1%}, "
              f"mayúsculas {self.char_distribution['uppercase']:.1%}")
        
    def _load_fonts(self) -> List:
        """Cargar fuentes con tamaños apropiados para 230x80"""
        fonts = []
        
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf", 
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/verdana.ttf",
            "fonts/font1.ttf"
        ]
        
        # Tamaños calibrados para dimensiones reales
        sizes = [32, 36, 40, 44, 48]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    for size in sizes:
                        fonts.append(ImageFont.truetype(font_path, size))
                except:
                    continue
        
        if not fonts:
            fonts = [ImageFont.load_default() for _ in range(5)]
            
        print(f"🔤 Fuentes optimizadas cargadas: {len(fonts)}")
        return fonts
    
    def generate_balanced_text(self) -> str:
        """Generar texto con distribución balanceada según análisis real"""
        text = ""
        
        for _ in range(self.captcha_length):
            rand = random.random()
            
            if rand < self.char_distribution['digits']:
                # Dígito (0-9)
                text += random.choice(string.digits)
            elif rand < self.char_distribution['digits'] + self.char_distribution['lowercase']:
                # Minúscula (a-z)
                text += random.choice(string.ascii_lowercase)
            else:
                # Mayúscula (A-Z)
                text += random.choice(string.ascii_uppercase)
        
        return text
    
    def add_realistic_noise(self, image: Image.Image) -> Image.Image:
        """Añadir ruido calibrado según CAPTCHAs reales"""
        img_array = np.array(image)
        
        # Ruido gaussiano calibrado
        noise = np.random.normal(0, self.noise_intensity * 255, img_array.shape)
        noisy_img = img_array.astype(np.float32) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        img_pil = Image.fromarray(noisy_img)
        draw = ImageDraw.Draw(img_pil)
        
        # Líneas de interferencia (calibradas: ~385 líneas reales)
        # Reducir proporcionalmente para generación eficiente
        num_lines = random.randint(15, 25)  # Escalado para eficiencia
        
        for _ in range(num_lines):
            # Líneas que atraviesan toda la imagen
            if random.random() < 0.7:  # 70% líneas horizontales/diagonales
                x1 = random.randint(0, self.width)
                y1 = random.randint(0, self.height)
                x2 = random.randint(0, self.width) 
                y2 = random.randint(0, self.height)
            else:  # 30% líneas más cortas
                x1 = random.randint(0, self.width//2)
                y1 = random.randint(0, self.height)
                x2 = x1 + random.randint(20, 60)
                y2 = y1 + random.randint(-10, 10)
            
            # Colores de líneas grises
            color = random.choice([
                (180, 180, 180),
                (160, 160, 160), 
                (200, 200, 200),
                (140, 140, 140)
            ])
            width = random.randint(1, 2)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        
        # Puntos aleatorios
        num_dots = random.randint(100, 200)
        for _ in range(num_dots):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            color = random.choice([
                (190, 190, 190),
                (170, 170, 170),
                (210, 210, 210)
            ])
            draw.point((x, y), fill=color)
        
        return img_pil
    
    def apply_realistic_distortion(self, image: Image.Image) -> Image.Image:
        """Aplicar distorsiones sutiles como en CAPTCHAs reales"""
        img_array = np.array(image)
        rows, cols = img_array.shape[:2]
        
        # Distorsión muy sutil (CAPTCHAs reales tienen poca distorsión)
        map_x = np.zeros((rows, cols), dtype=np.float32)
        map_y = np.zeros((rows, cols), dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
                # Ondulación muy ligera
                offset_x = 3 * np.sin(2 * np.pi * i / 40)
                offset_y = 2 * np.sin(2 * np.pi * j / 60)
                
                map_x[i, j] = j + offset_x
                map_y[i, j] = i + offset_y
        
        distorted = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)
        return Image.fromarray(distorted)
    
    def generate_single_captcha(self, text: str = None) -> Tuple[Image.Image, str]:
        """Generar CAPTCHA con parámetros optimizados"""
        if text is None:
            text = self.generate_balanced_text()
        
        # Fondo (mayormente blanco como en CAPTCHAs reales)
        bg_color = random.choice(self.bg_colors)
        image = Image.new('RGB', (self.width, self.height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Configuración de texto
        text_color = random.choice(self.text_colors)
        font = random.choice(self.fonts)
        
        # Posicionamiento optimizado para 230x80
        char_width = self.width // (self.captcha_length + 0.5)
        start_x = char_width // 2
        
        # Dibujar caracteres con espaciado real
        for i, char in enumerate(text):
            # Posición con variación menor (CAPTCHAs reales son más regulares)
            x = start_x + i * char_width + random.randint(-5, 5)
            y = self.height // 2 - 10 + random.randint(-3, 3)
            
            # Rotación mínima (CAPTCHAs reales tienen poca rotación)
            rotation = random.randint(-8, 8)
            
            # Crear carácter
            char_img = Image.new('RGBA', (60, 60), (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_img)
            char_draw.text((30, 30), char, font=font, fill=text_color, anchor="mm")
            
            # Rotar ligeramente
            if rotation != 0:
                char_img = char_img.rotate(rotation, expand=True)
            
            # Pegar en imagen
            paste_x = int(x - 30)
            paste_y = int(y - 30)
            image.paste(char_img, (paste_x, paste_y), char_img)
        
        # Aplicar efectos realistas
        if random.random() < 0.8:  # 80% ruido (más frecuente en CAPTCHAs reales)
            image = self.add_realistic_noise(image)
        
        if random.random() < 0.3:  # 30% distorsión ligera
            image = self.apply_realistic_distortion(image)
        
        # Blur muy ligero ocasional
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.3))
        
        return image, text
    
    def generate_dataset(self, num_samples: int, output_dir: str):
        """Generar dataset optimizado"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎯 Generando {num_samples} CAPTCHAs optimizados...")
        print(f"📁 Directorio: {output_dir}")
        
        generated = 0
        
        # Estadísticas para verificar distribución
        char_stats = {'digits': 0, 'lowercase': 0, 'uppercase': 0}
        
        while generated < num_samples:
            image, label = self.generate_single_captcha()
            
            # Guardar imagen
            filename = f"{label}_{generated:06d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            
            # Actualizar estadísticas
            for char in label:
                if char.isdigit():
                    char_stats['digits'] += 1
                elif char.islower():
                    char_stats['lowercase'] += 1
                elif char.isupper():
                    char_stats['uppercase'] += 1
            
            generated += 1
            
            if generated % 1000 == 0:
                print(f"  ✅ Generados: {generated}")
        
        # Mostrar estadísticas finales
        total_chars = sum(char_stats.values())
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"Total imágenes: {generated}")
        print(f"Total caracteres: {total_chars}")
        print(f"Dígitos: {char_stats['digits']/total_chars:.1%} (objetivo: {self.char_distribution['digits']:.1%})")
        print(f"Minúsculas: {char_stats['lowercase']/total_chars:.1%} (objetivo: {self.char_distribution['lowercase']:.1%})")
        print(f"Mayúsculas: {char_stats['uppercase']/total_chars:.1%} (objetivo: {self.char_distribution['uppercase']:.1%})")

def main():
    """Generar dataset optimizado más pequeño para pruebas rápidas"""
    generator = OptimizedCaptchaGenerator()
    
    # Dataset más pequeño para entrenamiento rápido
    train_samples = 10000  # 10k para entrenamiento rápido
    val_samples = 1000     # 1k para validación
    
    base_dir = "dataset_optimized"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    print("🚀 GENERACIÓN DE DATASET OPTIMIZADO")
    print("=" * 50)
    
    generator.generate_dataset(train_samples, train_dir)
    print("\n" + "="*30)
    generator.generate_dataset(val_samples, val_dir)
    
    print(f"\n🎉 DATASET OPTIMIZADO COMPLETO: {base_dir}")
    print(f"📈 Training: {train_samples} muestras")
    print(f"📊 Validation: {val_samples} muestras")
    print(f"🎯 Calibrado con parámetros reales de sssalud")

if __name__ == "__main__":
    main()
