#!/usr/bin/env python3
"""
Script de prueba para el módulo de visión
Autor: Sebastian Yambay
Repositorio: https://github.com/SebastianYambay/Proyecto_Base_Conocimiento
Uso: python test_vision.py <ruta_a_imagen>
"""

import sys
import os
from pathlib import Path

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.vision_module import analyze_water_turbidity

def test_vision_module(image_path):
    """Prueba el módulo de visión con una imagen"""
    
    print("🔬 Iniciando análisis de turbidez con OpenAI Vision API...")
    print(f"📸 Imagen: {image_path}\n")
    
    # Verificar que existe la imagen
    if not Path(image_path).exists():
        print(f"❌ Error: No se encontró la imagen en {image_path}")
        return
    
    # Leer imagen
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    print(f"✅ Imagen cargada: {len(image_bytes)} bytes\n")
    print("⏳ Enviando a OpenAI API (esto puede tomar 10-20 segundos)...\n")
    
    # Analizar
    result = analyze_water_turbidity(image_bytes)
    
    # Mostrar resultados
    if result.get('error'):
        print("❌ ERROR:")
        print(result['message'])
        return
    
    print("=" * 60)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 60)
    print(f"\n💧 TURBIDEZ: {result['ntu']} NTU")
    print(f"📊 CLASIFICACIÓN: {result['classification']}")
    print(f"🎯 CONFIANZA: {result['confidence']}%")
    print(f"✅ CUMPLE OMS: {'Sí' if result['meets_who_standards'] else 'No'}")
    
    print("\n" + "-" * 60)
    print("👁️  OBSERVACIONES VISUALES:")
    print("-" * 60)
    for key, value in result['color_profile'].items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "-" * 60)
    print("🔬 INDICADORES DE CALIDAD:")
    print("-" * 60)
    for key, value in result['ai_insights']['quality_indicators'].items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "-" * 60)
    print("🔍 POSIBLES CAUSAS:")
    print("-" * 60)
    for i, cause in enumerate(result['ai_insights']['potential_causes'], 1):
        print(f"  {i}. {cause}")
    
    print("\n" + "-" * 60)
    print("💡 RECOMENDACIÓN:")
    print("-" * 60)
    print(result['recommendation'])
    
    if result['ai_insights']['image_quality_notes']:
        print("\n" + "-" * 60)
        print("📸 NOTAS DE CALIDAD DE IMAGEN:")
        print("-" * 60)
        print(f"  {result['ai_insights']['image_quality_notes']}")
    
    print("\n" + "=" * 60)
    print(f"🤖 Powered by: {result['powered_by']}")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Uso: python test_vision.py <ruta_a_imagen>")
        print("\nEjemplo:")
        print("  python test_vision.py ./samples/water_sample.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_vision_module(image_path)
