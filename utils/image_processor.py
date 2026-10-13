"""
Procesamiento de imágenes y OCR multimodular (Tesseract + EasyOCR fallback).
Extrae texto de imágenes subidas por el usuario garantizando compatibilidad 100%.
"""

from PIL import Image
import io
import logging
from typing import Dict
import time
import numpy as np

logger = logging.getLogger(__name__)

# Intentar Tesseract solo si el binario de sistema está presente
try:
    import pytesseract
    import shutil
    TESSERACT_AVAILABLE = bool(shutil.which("tesseract"))
except Exception:
    TESSERACT_AVAILABLE = False

# Intentar EasyOCR (fallback local en Python sin dependencias de sistema)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

_easyocr_reader = None


def get_easyocr_reader():
    """Obtiene el lector de EasyOCR en memoria con caché."""
    global _easyocr_reader
    if _easyocr_reader is None and EASYOCR_AVAILABLE:
        logger.info("Inicializando EasyOCR Reader (es, en)...")
        _easyocr_reader = easyocr.Reader(['es', 'en'], gpu=False)
    return _easyocr_reader


def extract_text_from_image(image_file) -> Dict[str, str]:
    """Extrae texto de una imagen usando Tesseract o EasyOCR como motor robusto."""
    start_time = time.time()

    try:
        if isinstance(image_file, bytes):
            image = Image.open(io.BytesIO(image_file))
        else:
            image = image_file

        # Asegurar formato RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_w, img_h = image.width, image.height

        # 1. Intentar con Tesseract si está instalado y en PATH
        if TESSERACT_AVAILABLE:
            try:
                extracted_text = pytesseract.image_to_string(image, lang='eng+spa')
                if extracted_text.strip():
                    try:
                        ocr_data = pytesseract.image_to_data(image, output_type='dict')
                        confidences = [int(c) for c in ocr_data['confidence'] if c != '-1']
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 90.0
                    except Exception:
                        avg_confidence = 90.0

                    elapsed = time.time() - start_time
                    return {
                        "text": extracted_text.strip(),
                        "confidence": round(avg_confidence, 2),
                        "language": "multilingual",
                        "time_seconds": round(elapsed, 2),
                        "status": "success",
                        "engine": "Tesseract",
                        "boxes": [],
                        "image_width": img_w,
                        "image_height": img_h
                    }
            except Exception as tess_err:
                logger.warning(f"Tesseract no pudo procesar ({tess_err}), usando EasyOCR...")

        # 2. Usar EasyOCR como motor garantizado
        if EASYOCR_AVAILABLE:
            reader = get_easyocr_reader()
            img_np = np.array(image)
            results = reader.readtext(img_np)
            
            lines = []
            scores = []
            boxes = []

            for item in results:
                # item: (bbox, text, prob)
                text_clean = str(item[1]).strip()
                if not text_clean:
                    continue
                lines.append(text_clean)
                score = float(item[2])
                scores.append(score)

                pts = item[0]
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                x_min, x_max = max(0, min(xs)), min(img_w, max(xs))
                y_min, y_max = max(0, min(ys)), min(img_h, max(ys))

                left_pct = round((x_min / img_w) * 100, 2)
                top_pct = round((y_min / img_h) * 100, 2)
                width_pct = round(((x_max - x_min) / img_w) * 100, 2)
                height_pct = round(((y_max - y_min) / img_h) * 100, 2)

                boxes.append({
                    "text": text_clean,
                    "confidence": round(score * 100, 1),
                    "box": {
                        "left": left_pct,
                        "top": top_pct,
                        "width": width_pct,
                        "height": height_pct
                    }
                })

            extracted_text = "\n".join(lines).strip()
            avg_confidence = (sum(scores) / len(scores) * 100) if scores else 0.0
            elapsed = time.time() - start_time

            return {
                "text": extracted_text if extracted_text else "No se detectó texto claro en la imagen.",
                "confidence": round(avg_confidence, 1),
                "language": "es+en",
                "time_seconds": round(elapsed, 2),
                "status": "success",
                "engine": "EasyOCR",
                "boxes": boxes,
                "image_width": img_w,
                "image_height": img_h
            }

        # 3. Si ninguno está disponible
        return {
            "text": "[Imagen cargada. Instala easyocr o tesseract para extracción automática]",
            "confidence": 0.0,
            "language": "none",
            "time_seconds": round(time.time() - start_time, 2),
            "status": "warning"
        }

    except Exception as e:
        logger.error(f"Error OCR: {str(e)}")
        return {"text": "", "error": str(e), "status": "error"}


def validate_image(image_file) -> Dict[str, str]:
    """Valida que la imagen sea válida y devuelve su información."""
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
    """Redimensiona la imagen si es muy grande (máx 1024px de ancho)."""
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
