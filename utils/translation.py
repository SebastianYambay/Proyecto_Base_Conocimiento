"""
Traducción inglés <-> español usando MarianMT.
MarianMT es un Transformer encoder-decoder entrenado para traducción automática.
Optimizado con caché en memoria permanente para inferencia ultrarrápida.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from typing import Dict
import time

logger = logging.getLogger(__name__)

import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from typing import Dict
import time

logger = logging.getLogger(__name__)

USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

# Lock para evitar que dos hilos inicialicen el modelo simultáneamente
_model_lock = threading.Lock()
# Caché en memoria para no recargar los modelos en cada petición
_models_cache = {}


def get_model_and_tokenizer(model_name: str):
    """Carga y cachea el modelo y tokenizer en RAM/VRAM de forma segura y sin meta tensors."""
    with _model_lock:
        if model_name not in _models_cache:
            logger.info(f"Cargando y cacheando modelo en memoria: {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=False)
            if USE_CUDA:
                model = model.to("cuda")
            model.eval()
            _models_cache[model_name] = (tokenizer, model)
            logger.info(f"Modelo {model_name} cargado y listo.")
        return _models_cache[model_name]


def preload_translation_models():
    """Precarga ambos modelos al inicio del servidor para latencia cero."""
    try:
        get_model_and_tokenizer("Helsinki-NLP/opus-mt-en-es")
        get_model_and_tokenizer("Helsinki-NLP/opus-mt-es-en")
        logger.info("Todos los modelos de traducción han sido precargados con éxito.")
    except Exception as e:
        logger.error(f"Error precargando modelos de traducción: {e}")


def translate_en_to_es(text: str, max_length: int = 512) -> Dict[str, str]:
    """Traduce texto de inglés a español con MarianMT."""
    start_time = time.time()
    if not text.strip():
        return {"original": text, "translated": "", "status": "success", "time_seconds": 0}

    try:
        model_name = "Helsinki-NLP/opus-mt-en-es"
        tokenizer, model = get_model_and_tokenizer(model_name)

        # Limitar largo del texto
        text = text[:max_length * 4]

        # Tokenizar y generar traducción
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        if USE_CUDA:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, num_beams=4)

        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start_time

        return {
            "original": text,
            "translated": translated_text,
            "source_lang": "en",
            "target_lang": "es",
            "time_seconds": round(elapsed, 2),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error en traducción EN->ES: {str(e)}")
        _models_cache.pop("Helsinki-NLP/opus-mt-en-es", None)
        return {"original": text, "translated": "", "error": str(e), "status": "error"}


def translate_es_to_en(text: str, max_length: int = 512) -> Dict[str, str]:
    """Traduce texto de español a inglés con MarianMT."""
    start_time = time.time()
    if not text.strip():
        return {"original": text, "translated": "", "status": "success", "time_seconds": 0}

    try:
        model_name = "Helsinki-NLP/opus-mt-es-en"
        tokenizer, model = get_model_and_tokenizer(model_name)

        text = text[:max_length * 4]

        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        if USE_CUDA:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, num_beams=4)

        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start_time

        return {
            "original": text,
            "translated": translated_text,
            "source_lang": "es",
            "target_lang": "en",
            "time_seconds": round(elapsed, 2),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error en traducción ES->EN: {str(e)}")
        return {"original": text, "translated": "", "error": str(e), "status": "error"}


def translate(text: str, source_lang: str = "en", target_lang: str = "es") -> Dict:
    """Función principal de traducción. Redirige al modelo correcto según idioma."""
    if source_lang == "en" and target_lang == "es":
        return translate_en_to_es(text)
    elif source_lang == "es" and target_lang == "en":
        return translate_es_to_en(text)
    else:
        return {
            "status": "error",
            "error": f"Par {source_lang}-{target_lang} no soportado. Usa en-es o es-en."
        }
