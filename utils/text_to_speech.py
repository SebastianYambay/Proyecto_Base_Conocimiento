"""
Texto a Audio con Facebook MMS-TTS.
Soporta ingles y espanol. El pipeline genera audio WAV a 16kHz.
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

# Cache manual de pipelines (evita recargar el modelo cada vez)
_tts_pipelines = {}


def _get_tts_pipeline(lang: str):
    """Devuelve el pipeline TTS para el idioma pedido. Lo cachea en memoria."""
    if lang not in TTS_MODELS:
        raise ValueError(f"Idioma '{lang}' no soportado. Disponibles: {list(TTS_MODELS.keys())}")

    if lang not in _tts_pipelines:
        model_name = TTS_MODELS[lang]
        logger.info(f"Cargando modelo TTS: {model_name}")
        _tts_pipelines[lang] = pipeline(
            "text-to-speech",
            model=model_name,
            device=0 if DEVICE == "cuda" else -1
        )

    return _tts_pipelines[lang]


def text_to_speech(text: str, lang: str = "en") -> Dict:
    """Convierte texto a audio WAV. Devuelve bytes listos para st.audio()."""
    start_time = time.time()

    try:
        pipe = _get_tts_pipeline(lang)
        result = pipe(text)

        audio_array = np.array(result["audio"])
        sampling_rate = result["sampling_rate"]

        # Convertir a WAV: normalizar a int16 y escribir en buffer
        audio_buffer = io.BytesIO()
        audio_normalized = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
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
