"""
Módulo de Visión y Análisis de Turbidez
Autor: Sebastian Yambay
Repositorio: https://github.com/SebastianYambay/Proyecto_Base_Conocimiento
"""

import base64
import io
import json
import os
import re
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ---------------------------------------------------------
# Configuración del LLM local de visión (servidor compatible con OpenAI).
# El análisis de turbidez requiere un modelo MULTIMODAL (visión), por ejemplo
# llava, qwen2-vl o moondream cargado en LM Studio / Ollama.
# Un modelo solo-texto como mistral-7b-instruct NO puede analizar imágenes.
# ---------------------------------------------------------
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "mistral-7b-instruct-v0.1")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "lm-studio")

# Cliente apuntando al servidor local (sin consumir la API de OpenAI)
client = OpenAI(base_url=LOCAL_LLM_BASE_URL, api_key=LOCAL_LLM_API_KEY)

# ---------------------------------------------------------
# Prompts del sistema
# ---------------------------------------------------------
SYSTEM_PROMPT = """
Eres un analista experto en calidad del agua con profundo conocimiento en evaluación de turbidez e indicadores visuales de calidad del agua.

Tu tarea es analizar muestras de agua y estimar la turbidez en NTU (Unidades Nefelométricas de Turbidez) basándote en la apariencia visual.

Consideraciones clave:
- Esta es una ESTIMACIÓN VISUAL a partir de una imagen, NO una medición de laboratorio
- Enfócate en claridad, partículas suspendidas, color, dispersión de luz y transparencia general
- Considera las condiciones de iluminación y calidad de imagen en tu evaluación de confianza
- Ignora reflejos, brillos o artefactos del contenedor a menos que afecten claramente la visibilidad del agua
- Proporciona observaciones detalladas que justifiquen tu estimación de NTU

Requisitos de salida:
- Devuelve SOLO JSON válido (sin markdown, sin bloques de código, sin texto extra)
- Sé preciso y técnicamente exacto
- Proporciona información útil y accionable para operadores de plantas de tratamiento
- IMPORTANTE: Responde TODOS los campos en ESPAÑOL
"""

USER_PROMPT = """
Analiza esta imagen de muestra de agua y proporciona una evaluación completa de turbidez.

Usa estas referencias visuales para la estimación de NTU:

📊 Escala NTU:
- 0-1 NTU: Cristalina, completamente transparente, sin partículas visibles, claridad excelente
- 1-5 NTU: Muy clara, estándar OMS para agua potable, neblina mínima
- 5-10 NTU: Ligera turbiedad, visible bajo luz, partículas apenas perceptibles
- 10-25 NTU: Turbiedad notable, partículas suspendidas visibles, claridad reducida
- 25-50 NTU: Claramente turbia, muchas partículas visibles, objetos de fondo oscurecidos
- 50-100 NTU: Muy turbia, apariencia opaca, sedimento pesado, calidad pobre
- >100 NTU: Extremadamente turbia, completamente opaca, calidad inaceptable

Proporciona tu análisis en este formato JSON exacto (TODOS LOS TEXTOS EN ESPAÑOL):
{
  "turbidity_ntu": <número entre 0 y 150>,
  "confidence_score": <porcentaje 0-100 basado en calidad de imagen y claridad de indicadores>,
  "visual_observations": {
    "clarity": "<descripción breve de la claridad del agua en español>",
    "color_tint": "<color presente: transparente/amarillento/marrón/verdoso/etc en español>",
    "visible_particles": "<ninguna/pocas/moderadas/muchas/abundantes en español>",
    "light_transmission": "<excelente/buena/aceptable/pobre en español>"
  },
  "quality_indicators": {
    "suspended_solids": "<bajo/medio/alto en español>",
    "sediment_presence": "<ninguno/mínimo/moderado/abundante en español>",
    "organic_matter": "<no visible/posiblemente presente/claramente presente en español>"
  },
  "treatment_recommendations": [
    "<acción específica 1 en español, máximo 100 caracteres>",
    "<acción específica 2 en español, máximo 100 caracteres>"
  ],
  "potential_causes": [
    "<causa probable 1 en español, máximo 80 caracteres>",
    "<causa probable 2 en español, máximo 80 caracteres>"
  ],
  "image_quality_notes": "<notas sobre iluminación, enfoque o calidad de imagen en español, máximo 150 caracteres>"
}

IMPORTANTE: Todos los textos descriptivos deben estar en ESPAÑOL y ser concisos para mostrarse correctamente en la interfaz.
"""

# Prompt simple para modelos compactos (moondream) que no siguen JSON
SIMPLE_NTU_PROMPT = (
    "Observa esta muestra de agua y estima su turbidez en NTU "
    "(escala de 0 a 150, donde 0 es cristalina y 150 muy turbia). "
    "Responde ÚNICAMENTE con un número."
)


def _prepare_image_base64(image_bytes, max_side=1024):
    """
    Redimensiona la imagen a un lado máximo y la re-codifica como JPEG antes de
    enviarla al modelo de visión. Las imágenes grandes consumen muchísimos
    tokens (qwen2-vl tokeniza la imagen completa) y pueden exceder el contexto.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        # Si PIL falla por algún formato raro, usar los bytes originales
        return base64.b64encode(image_bytes).decode("utf-8")


def _classify_ntu(ntu_value):
    """Devuelve (clasificación, status) según el valor NTU."""
    if ntu_value < 1:
        return "Excelente", "safe"
    elif ntu_value < 5:
        return "Muy Buena", "safe"
    elif ntu_value < 10:
        return "Buena", "acceptable"
    elif ntu_value < 25:
        return "Aceptable", "acceptable"
    elif ntu_value < 50:
        return "Deficiente", "poor"
    else:
        return "Muy Turbia", "poor"


def _extract_first_number(text):
    """Extrae el primer número (entero o decimal) de un texto, o None."""
    if not text:
        return None
    m = re.search(r'\d+(?:\.\d+)?', text)
    return float(m.group()) if m else None


def _fallback_simple_analysis(image_base64, raw_text):
    """
    Modo de respaldo para modelos que no devuelven JSON (ej. moondream):
    hace una pregunta enfocada para obtener solo el número NTU y arma el
    resultado completo. Devuelve None si no logra estimar un número.
    """
    num_text = None
    try:
        resp = client.chat.completions.create(
            model=LOCAL_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": SIMPLE_NTU_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }],
            temperature=0.2,
        )
        num_text = resp.choices[0].message.content.strip()
    except Exception:
        pass

    ntu_value = _extract_first_number(num_text)
    if ntu_value is None:
        ntu_value = _extract_first_number(raw_text)
    if ntu_value is None:
        return None  # no se pudo estimar un valor

    ntu_value = max(0.0, min(ntu_value, 150.0))
    classification, status = _classify_ntu(ntu_value)

    desc = (raw_text or '').strip()
    desc_short = desc if len(desc) <= 120 else desc[:117] + '...'

    return {
        'ntu': round(ntu_value, 2),
        'classification': classification,
        'status': status,
        'confidence': 55,  # estimación de un modelo compacto
        'color_profile': {
            'clarity': desc_short or 'Estimación visual',
            'color_tint': 'N/D',
            'visible_particles': 'N/D',
            'light_transmission': 'N/D',
        },
        'recommendation': get_recommendation(ntu_value),
        'meets_who_standards': ntu_value < 5,
        'ai_insights': {
            'quality_indicators': {'suspended_solids': 'N/D', 'sediment_presence': 'N/D', 'organic_matter': 'N/D'},
            'potential_causes': [],
            'image_quality_notes': 'Estimación de un modelo compacto (moondream); puede ser aproximada.',
        },
        'powered_by': f'LLM Local ({LOCAL_VISION_MODEL}) · modo simple',
    }


# ---------------------------------------------------------
# Función principal mejorada
# ---------------------------------------------------------
def analyze_water_turbidity(image_bytes):
    """
    Analiza la turbidez del agua usando un modelo de visión local
    (servidor compatible con OpenAI: LM Studio / Ollama).

    Args:
        image_bytes: Bytes de la imagen subida
        
    Returns:
        dict: Análisis completo de turbidez con métricas detalladas
    """
    try:
        # Redimensionar y convertir a base64 (controla el consumo de tokens)
        image_base64 = _prepare_image_base64(image_bytes)

        # Llamada al modelo de visión local (servidor compatible con OpenAI)
        response = client.chat.completions.create(
            model=LOCAL_VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.3  # Baja temperatura para respuestas más consistentes
        )
        
        # Extraer respuesta
        raw_output = response.choices[0].message.content.strip()
        
        # Limpiar markdown si existe
        if raw_output.startswith('```json'):
            raw_output = raw_output.replace('```json', '').replace('```', '').strip()
        elif raw_output.startswith('```'):
            raw_output = raw_output.replace('```', '').strip()

        # Extraer el bloque JSON aunque el modelo agregue texto antes/después
        start = raw_output.find('{')
        end = raw_output.rfind('}')
        json_str = raw_output[start:end + 1] if (start != -1 and end > start) else raw_output

        # Parsear JSON
        ai_analysis = json.loads(json_str)
        
        # Extraer valores principales
        ntu_value = float(ai_analysis.get('turbidity_ntu', 0))
        confidence = int(ai_analysis.get('confidence_score', 75))
        
        # Clasificación basada en NTU
        if ntu_value < 1:
            classification = "Excelente"
            status = "safe"
        elif ntu_value < 5:
            classification = "Muy Buena"
            status = "safe"
        elif ntu_value < 10:
            classification = "Buena"
            status = "acceptable"
        elif ntu_value < 25:
            classification = "Aceptable"
            status = "acceptable"
        elif ntu_value < 50:
            classification = "Deficiente"
            status = "poor"
        else:
            classification = "Muy Turbia"
            status = "poor"
        
        # Construir perfil visual para compatibilidad con UI
        visual_obs = ai_analysis.get('visual_observations', {})
        
        # Función auxiliar para truncar texto
        def truncate_text(text, max_length=50):
            """Trunca texto largo para mejor visualización en UI"""
            if not text or text == 'No disponible':
                return text
            text = str(text).strip()
            return text if len(text) <= max_length else text[:max_length-3] + '...'
        
        color_profile = {
            'clarity': truncate_text(visual_obs.get('clarity', 'No disponible')),
            'color_tint': truncate_text(visual_obs.get('color_tint', 'No disponible'), 30),
            'visible_particles': truncate_text(visual_obs.get('visible_particles', 'No disponible'), 20),
            'light_transmission': truncate_text(visual_obs.get('light_transmission', 'No disponible'), 20)
        }
        
        # Construir recomendación
        recommendations = ai_analysis.get('treatment_recommendations', [])
        recommendation_text = get_recommendation(ntu_value)
        if recommendations:
            # Limitar a 2 recomendaciones principales
            top_recommendations = recommendations[:2]
            recommendation_text += "\n\n🔧 Recomendaciones AI:\n" + "\n".join(f"• {r}" for r in top_recommendations)
        
        # Procesar quality indicators con valores truncados
        quality_indicators = ai_analysis.get('quality_indicators', {})
        quality_indicators_clean = {
            'suspended_solids': truncate_text(quality_indicators.get('suspended_solids', 'N/A'), 20),
            'sediment_presence': truncate_text(quality_indicators.get('sediment_presence', 'N/A'), 20),
            'organic_matter': truncate_text(quality_indicators.get('organic_matter', 'N/A'), 30)
        }
        
        # Procesar causas potenciales
        potential_causes = ai_analysis.get('potential_causes', [])
        potential_causes_clean = [truncate_text(cause, 100) for cause in potential_causes[:3]]
        
        # Notas de calidad de imagen
        image_notes = truncate_text(ai_analysis.get('image_quality_notes', ''), 200)
        
        return {
            'ntu': round(ntu_value, 2),
            'classification': classification,
            'status': status,
            'confidence': confidence,
            'color_profile': color_profile,
            'recommendation': recommendation_text,
            'meets_who_standards': ntu_value < 5,
            'ai_insights': {
                'quality_indicators': quality_indicators_clean,
                'potential_causes': potential_causes_clean,
                'image_quality_notes': image_notes
            },
            'powered_by': f'LLM Local ({LOCAL_VISION_MODEL})'
        }

    except json.JSONDecodeError as e:
        # El modelo no devolvió JSON (típico en modelos compactos como moondream).
        # Intentar modo de respaldo: pedir solo el número NTU y armar el resultado.
        fallback = _fallback_simple_analysis(image_base64, raw_output)
        if fallback is not None:
            return fallback

        # Si ni el respaldo logró un número, reportar el error con preview
        preview = raw_output[:300] if len(raw_output) > 300 else raw_output
        return {
            'error': True,
            'message': (
                f"❌ El modelo de visión no devolvió un valor de turbidez utilizable.\n\n"
                f"Prueba con un modelo multimodal que siga instrucciones mejor "
                f"(qwen2-vl, llava).\n\n"
                f"Error: {str(e)}\n\nRespuesta recibida:\n{preview}"
            ),
            'ntu': None
        }
    except Exception as e:
        error_msg = str(e)
        # Mensajes de error más amigables
        if 'connection' in error_msg.lower() or 'timeout' in error_msg.lower() or 'connect' in error_msg.lower():
            friendly_msg = (
                f"🌐 No se pudo contactar al servidor local en {LOCAL_LLM_BASE_URL}. "
                f"Verifica que LM Studio / Ollama esté corriendo."
            )
        elif 'model' in error_msg.lower() or '404' in error_msg:
            friendly_msg = (
                f"🔍 El modelo '{LOCAL_VISION_MODEL}' no está disponible o no soporta "
                f"visión. Carga un modelo multimodal (llava, qwen2-vl, moondream)."
            )
        else:
            friendly_msg = f"⚠️ Error en análisis con el LLM local: {error_msg}"

        return {
            'error': True,
            'message': friendly_msg,
            'ntu': None
        }

# ---------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------
def get_recommendation(ntu_value):
    """
    Proporciona recomendaciones basadas en el valor NTU.
    """
    if ntu_value < 1:
        return "💎 Calidad excepcional. Agua cristalina."
    elif ntu_value < 5:
        return "✅ Cumple con estándares de la OMS. Apta para consumo."
    elif ntu_value < 10:
        return "⚠️ Aceptable pero considere filtración adicional."
    elif ntu_value < 25:
        return "🔶 Requiere tratamiento antes del consumo."
    elif ntu_value < 50:
        return "🚫 No apta para consumo. Requiere tratamiento intensivo."
    else:
        return "❌ Turbidez extremadamente alta. Tratamiento crítico requerido."

def get_ntu_interpretation():
    """
    Retorna información educativa sobre NTU.
    """
    return {
        'title': 'Escala de Turbidez (NTU)',
        'ranges': [
            {'range': '0-1 NTU', 'quality': 'Excelente', 'description': 'Agua cristalina'},
            {'range': '1-5 NTU', 'quality': 'Muy Buena', 'description': 'Estándar OMS'},
            {'range': '5-10 NTU', 'quality': 'Buena', 'description': 'Ligeramente visible'},
            {'range': '10-25 NTU', 'quality': 'Aceptable', 'description': 'Visible, requiere atención'},
            {'range': '25-50 NTU', 'quality': 'Deficiente', 'description': 'No apta para consumo'},
            {'range': '>50 NTU', 'quality': 'Muy Turbia', 'description': 'Calidad crítica'}
        ]
    }
