"""
Procesamiento de imagenes y OCR con Tesseract.
Extrae texto de imagenes subidas por el usuario.
"""

from PIL import Image
import io
import logging
from typing import Dict
import time

logger = logging.getLogger(__name__)

# Tesseract es opcional, si no esta instalado se usa un placeholder
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract no disponible. Instalar con: pip install pytesseract")


def extract_text_from_image(image_file) -> Dict[str, str]:
    """Extrae texto de una imagen usando OCR (Tesseract)."""
    start_time = time.time()

    try:
        if isinstance(image_file, bytes):
            image = Image.open(io.BytesIO(image_file))
        else:
            image = image_file

        # Sin Tesseract, devolver mensaje informativo
        if not TESSERACT_AVAILABLE:
            return {
                "text": "[Imagen cargada. Instala pytesseract para OCR.]",
                "confidence": 0.0,
                "language": "eng",
                "time_seconds": round(time.time() - start_time, 2),
                "status": "warning",
                "message": "Tesseract no instalado. Usa: sudo apt-get install tesseract-ocr"
            }

        # Extraer texto
        extracted_text = pytesseract.image_to_string(image, lang='eng')

        # Calcular confianza promedio
        try:
            ocr_data = pytesseract.image_to_data(image, output_type='dict')
            confidences = [int(c) for c in ocr_data['confidence'] if c != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except:
            avg_confidence = 0

        elapsed = time.time() - start_time

        return {
            "text": extracted_text.strip(),
            "confidence": round(avg_confidence, 2),
            "language": "eng",
            "time_seconds": round(elapsed, 2),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error OCR: {str(e)}")
        return {"text": "", "error": str(e), "status": "error"}


def validate_image(image_file) -> Dict[str, str]:
    """Valida que la imagen sea valida y devuelve su info."""
    try:
        if isinstance(image_file, bytes):
            image = Image.open(io.BytesIO(image_file))
        else:
            image = image_file

        return {
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
            "status": "valid"
        }

    except Exception as e:
        logger.error(f"Error validando imagen: {str(e)}")
        return {"status": "invalid", "error": str(e)}


def resize_image(image_file, max_width: int = 1024) -> Image.Image:
    """Redimensiona la imagen si es muy grande (max 1024px de ancho)."""
    if isinstance(image_file, bytes):
        image = Image.open(io.BytesIO(image_file))
    else:
        image = image_file

    if image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"Imagen redimensionada a {max_width}x{new_height}")

    return image
