"""
Carga y cache de modelos Transformer.
Usa st.cache_resource para que los modelos se carguen una sola vez.
"""

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Detecta si hay GPU disponible
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_translation_model(model_name: str = "Helsinki-NLP/opus-mt-en-es") -> Tuple:
    """Carga el modelo MarianMT para traduccion (se cachea automaticamente)."""
    logger.info(f"Cargando modelo de traduccion: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_translation_pipeline(
    source_lang: str = "en",
    target_lang: str = "es"
) -> Tuple[str, object]:
    """Carga el pipeline de traduccion segun el par de idiomas."""
    # Mapeo de idiomas a modelos de HuggingFace
    model_map = {
        ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
        ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    }

    model_name = model_map.get((source_lang, target_lang))
    if not model_name:
        raise ValueError(f"Par {source_lang}-{target_lang} no soportado")

    logger.info(f"Cargando pipeline: {model_name}")
    pipe = pipeline("translation", model=model_name, device=0 if DEVICE == "cuda" else -1)
    return model_name, pipe


def get_device_info() -> Dict[str, str]:
    """Devuelve info sobre el dispositivo (CPU/GPU)."""
    info = {
        "device": DEVICE,
        "cuda_available": str(torch.cuda.is_available()),
    }

    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"

    return info
