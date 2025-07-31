#!/usr/bin/env python3
"""
Analizador de CAPTCHAs reales para extraer características
Mejora el generador sintético basándose en imágenes reales
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
from collections import Counter

class RealCaptchaAnalyzer:
    def __init__(self, captcha_dir: str):
        self.captcha_dir = captcha_dir
        self.captcha_files = [f for f in os.listdir(captcha_dir) if f.endswith('.png')]
        print(f"📁 Analizando {len(self.captcha_files)} CAPTCHAs reales")
    
    def analyze_dimensions(self):
        """Analizar dimensiones de las imágenes"""
        print("\n📐 ANÁLISIS DE DIMENSIONES:")
        
        dimensions = []
        for filename in self.captcha_files:
            filepath = os.path.join(self.captcha_dir, filename)
            img = Image.open(filepath)
            dimensions.append(img.size)  # (width, height)
        
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        
        print(f"Ancho - Min: {min(widths)}, Max: {max(widths)}, Promedio: {np.mean(widths):.1f}")
        print(f"Alto - Min: {min(heights)}, Max: {max(heights)}, Promedio: {np.mean(heights):.1f}")
        print(f"Dimensión más común: {Counter(dimensions).most_common(1)[0]}")
        
        return {
            'avg_width': np.mean(widths),
            'avg_height': np.mean(heights),
            'most_common': Counter(dimensions).most_common(1)[0][0]
        }
    
    def analyze_colors(self):
        """Analizar colores predominantes"""
        print("\n🎨 ANÁLISIS DE COLORES:")
        
        bg_colors = []
        text_colors = []
        
        for filename in self.captcha_files:
            filepath = os.path.join(self.captcha_dir, filename)
            img = cv2.imread(filepath)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Análisis de color de fondo (esquinas)
            h, w = img_rgb.shape[:2]
            corners = [
                img_rgb[0, 0],           # Esquina superior izquierda
                img_rgb[0, w-1],         # Esquina superior derecha
                img_rgb[h-1, 0],         # Esquina inferior izquierda
                img_rgb[h-1, w-1]        # Esquina inferior derecha
            ]
            avg_bg_color = np.mean(corners, axis=0)
            bg_colors.append(tuple(avg_bg_color.astype(int)))
            
            # Análisis de color de texto (píxeles más oscuros)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            dark_threshold = np.percentile(gray, 20)  # 20% más oscuro
            dark_pixels = img_rgb[gray < dark_threshold]
            
            if len(dark_pixels) > 0:
                avg_text_color = np.mean(dark_pixels, axis=0)
                text_colors.append(tuple(avg_text_color.astype(int)))
        
        # Colores de fondo más comunes
        bg_counter = Counter(bg_colors)
        text_counter = Counter(text_colors)
        
        print("Colores de fondo más comunes:")
        for color, count in bg_counter.most_common(3):
            print(f"  RGB{color}: {count} veces")
        
        print("Colores de texto más comunes:")
        for color, count in text_counter.most_common(3):
            print(f"  RGB{color}: {count} veces")
        
        return {
            'bg_colors': [color for color, _ in bg_counter.most_common(5)],
            'text_colors': [color for color, _ in text_counter.most_common(5)]
        }
    
    def analyze_noise_patterns(self):
        """Analizar patrones de ruido y distorsión"""
        print("\n🔍 ANÁLISIS DE RUIDO:")
        
        noise_levels = []
        line_counts = []
        
        for filename in self.captcha_files:
            filepath = os.path.join(self.captcha_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            # Calcular nivel de ruido usando desviación estándar
            noise_level = np.std(img)
            noise_levels.append(noise_level)
            
            # Detectar líneas usando transformada de Hough
            edges = cv2.Canny(img, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
            line_count = len(lines) if lines is not None else 0
            line_counts.append(line_count)
        
        print(f"Nivel de ruido promedio: {np.mean(noise_levels):.1f}")
        print(f"Líneas de interferencia promedio: {np.mean(line_counts):.1f}")
        
        return {
            'avg_noise': np.mean(noise_levels),
            'avg_lines': np.mean(line_counts)
        }
    
    def analyze_text_characteristics(self):
        """Analizar características del texto"""
        print("\n📝 ANÁLISIS DE TEXTO:")
        
        # Análisis conocido de las etiquetas
        known_labels = [
            "UKhGh9", "26WanS", "e4TkHP", "cGfFE2", "gnRYZe",
            "v76Ebu", "DUzp49", "MWR3mw", "3h2vUF", "t2md2m"
        ]
        
        all_chars = ''.join(known_labels)
        char_counter = Counter(all_chars)
        
        # Análisis por tipo
        digits = sum(1 for c in all_chars if c.isdigit())
        lowercase = sum(1 for c in all_chars if c.islower())
        uppercase = sum(1 for c in all_chars if c.isupper())
        
        print(f"Total caracteres analizados: {len(all_chars)}")
        print(f"Dígitos: {digits} ({digits/len(all_chars)*100:.1f}%)")
        print(f"Minúsculas: {lowercase} ({lowercase/len(all_chars)*100:.1f}%)")
        print(f"Mayúsculas: {uppercase} ({uppercase/len(all_chars)*100:.1f}%)")
        
        print("\nCaracteres más frecuentes:")
        for char, count in char_counter.most_common(10):
            print(f"  '{char}': {count} veces")
        
        return {
            'digit_ratio': digits/len(all_chars),
            'lowercase_ratio': lowercase/len(all_chars),
            'uppercase_ratio': uppercase/len(all_chars),
            'char_frequencies': dict(char_counter)
        }
    
    def generate_optimization_suggestions(self):
        """Generar sugerencias para mejorar el generador"""
        print("\n🎯 EJECUTANDO ANÁLISIS COMPLETO...")
        
        dim_analysis = self.analyze_dimensions()
        color_analysis = self.analyze_colors()
        noise_analysis = self.analyze_noise_patterns()
        text_analysis = self.analyze_text_characteristics()
        
        print("\n" + "="*60)
        print("💡 SUGERENCIAS PARA EL GENERADOR:")
        print("="*60)
        
        print(f"📐 Dimensiones óptimas: {dim_analysis['most_common']}")
        
        print("🎨 Colores recomendados:")
        print(f"  Fondos: {color_analysis['bg_colors'][:3]}")
        print(f"  Texto: {color_analysis['text_colors'][:3]}")
        
        print("🔍 Parámetros de ruido:")
        print(f"  Intensidad: {noise_analysis['avg_noise']/255:.2f}")
        print(f"  Líneas interferencia: {int(noise_analysis['avg_lines'])}")
        
        print("📝 Distribución de caracteres:")
        print(f"  Dígitos: {text_analysis['digit_ratio']:.1%}")
        print(f"  Minúsculas: {text_analysis['lowercase_ratio']:.1%}")
        print(f"  Mayúsculas: {text_analysis['uppercase_ratio']:.1%}")
        
        # Generar código de configuración
        print("\n🔧 CÓDIGO PARA CONFIGURACIÓN:")
        print("-" * 40)
        print(f"WIDTH = {dim_analysis['most_common'][0]}")
        print(f"HEIGHT = {dim_analysis['most_common'][1]}")
        print(f"BG_COLORS = {color_analysis['bg_colors'][:3]}")
        print(f"TEXT_COLORS = {color_analysis['text_colors'][:3]}")
        print(f"NOISE_INTENSITY = {noise_analysis['avg_noise']/255:.2f}")
        print(f"AVG_LINES = {int(noise_analysis['avg_lines'])}")
        
        return {
            'dimensions': dim_analysis,
            'colors': color_analysis,
            'noise': noise_analysis,
            'text': text_analysis
        }

def main():
    """Función principal"""
    captcha_dir = "my_captchas"
    
    if not os.path.exists(captcha_dir):
        print(f"❌ Directorio no encontrado: {captcha_dir}")
        return
    
    analyzer = RealCaptchaAnalyzer(captcha_dir)
    results = analyzer.generate_optimization_suggestions()
    
    print(f"\n✅ Análisis completado!")
    print(f"📊 Usa estos parámetros para optimizar advanced_captcha_generator.py")

if __name__ == "__main__":
    main()
