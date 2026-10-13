"""
Texto a Audio con Facebook MMS-TTS.
Soporta inglés y español. El pipeline genera audio WAV a 16kHz.
Optimizado con caché en memoria permanente para síntesis instantánea.
"""

import torch
import numpy as np
from transformers import pipeline
import logging
import time
import io
import scipy.io.wavfile as wavfile
from typing import Dict

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modelos por idioma
TTS_MODELS = {
    "en": "facebook/mms-tts-eng",
    "es": "facebook/mms-tts-spa",
}

# Caché en memoria permanente
_tts_pipelines = {}


def _get_tts_pipeline(lang: str):
    """Devuelve el pipeline TTS para el idioma pedido. Lo cachea en memoria permanentemente."""
    if lang not in TTS_MODELS:
        raise ValueError(f"Idioma '{lang}' no soportado. Disponibles: {list(TTS_MODELS.keys())}")

    if lang not in _tts_pipelines:
        model_name = TTS_MODELS[lang]
        logger.info(f"Cargando modelo TTS en memoria: {model_name}...")
        _tts_pipelines[lang] = pipeline(
            "text-to-speech",
            model=model_name,
            device=0 if DEVICE == "cuda" else -1
        )
        logger.info(f"Modelo TTS {model_name} cargado y listo.")

    return _tts_pipelines[lang]


def preload_tts_models():
    """Precarga los pipelines TTS en memoria al iniciar el servidor."""
    try:
        _get_tts_pipeline("es")
        _get_tts_pipeline("en")
        logger.info("Todos los modelos TTS han sido precargados con éxito.")
    except Exception as e:
        logger.error(f"Error precargando TTS: {e}")


def text_to_speech(text: str, lang: str = "es") -> Dict:
    """Convierte texto a audio WAV. Devuelve bytes listos para reproducir."""
    start_time = time.time()
    if not text.strip():
        return {"audio_bytes": None, "text": text, "status": "error", "error": "Texto vacío"}

    try:
        pipe = _get_tts_pipeline(lang)
        result = pipe(text)

        audio_array = np.array(result["audio"])
        sampling_rate = result["sampling_rate"]

        # Convertir a WAV: normalizar a int16 y escribir en buffer
        audio_buffer = io.BytesIO()
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_normalized = np.int16(audio_array / max_val * 32767)
        else:
            audio_normalized = np.zeros_like(audio_array, dtype=np.int16)

        wavfile.write(audio_buffer, sampling_rate, audio_normalized)
        audio_bytes = audio_buffer.getvalue()

        elapsed = time.time() - start_time

        return {
            "audio_bytes": audio_bytes,
            "sampling_rate": sampling_rate,
            "text": text,
            "text_length": len(text),
            "lang": lang,
            "model": TTS_MODELS[lang],
            "time_seconds": round(elapsed, 2),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error TTS: {str(e)}")
        return {"audio_bytes": None, "text": text, "error": str(e), "status": "error"}
