"""
Traduccion ingles <-> espanol usando MarianMT.
MarianMT es un Transformer encoder-decoder entrenado para traduccion automatica.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from typing import Dict
import time

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def translate_en_to_es(text: str, max_length: int = 512) -> Dict[str, str]:
    """Traduce texto de ingles a espanol con MarianMT."""
    start_time = time.time()

    try:
        model_name = "Helsinki-NLP/opus-mt-en-es"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        model.eval()

        # Limitar largo del texto
        text = text[:max_length * 4]

        # Tokenizar y generar traduccion
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)

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
        logger.error(f"Error en traduccion: {str(e)}")
        return {"original": text, "translated": "", "error": str(e), "status": "error"}


def translate_es_to_en(text: str, max_length: int = 512) -> Dict[str, str]:
    """Traduce texto de espanol a ingles con MarianMT."""
    start_time = time.time()

    try:
        model_name = "Helsinki-NLP/opus-mt-es-en"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        model.eval()

        text = text[:max_length * 4]

        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)

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
        logger.error(f"Error en traduccion: {str(e)}")
        return {"original": text, "translated": "", "error": str(e), "status": "error"}


def translate(text: str, source_lang: str = "en", target_lang: str = "es") -> Dict:
    """Funcion principal de traduccion. Redirige al modelo correcto segun idioma."""
    if source_lang == "en" and target_lang == "es":
        return translate_en_to_es(text)
    elif source_lang == "es" and target_lang == "en":
        return translate_es_to_en(text)
    else:
        return {
            "status": "error",
            "error": f"Par {source_lang}-{target_lang} no soportado. Usa en-es o es-en."
        }
