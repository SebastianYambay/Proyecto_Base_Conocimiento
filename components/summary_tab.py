"""Pestana Texto a Audio: convierte texto a voz con MMS-TTS."""

import streamlit as st
from utils.text_to_speech import text_to_speech


def render_tts_tab():
    """Interfaz para generar audio a partir de texto."""
    st.markdown("### 🔊 Texto a Audio")
    st.markdown("Convierte texto a voz usando modelos MMS-TTS (Facebook)")

    # Selector de idioma
    lang_option = st.radio(
        "Idioma del texto:",
        ["Inglés (en)", "Español (es)"],
        horizontal=True,
        key="tts_lang"
    )
    lang_code = "en" if "en" in lang_option else "es"

    text_input = st.text_area(
        "Escribe el texto a convertir en audio:",
        placeholder="Type or paste your text here..." if lang_code == "en"
        else "Escribí o pegá tu texto acá...",
        height=200,
        key="tts_input"
    )

    if not text_input:
        return

    st.caption(f"📝 {len(text_input)} caracteres")

    if st.button("🔊 Generar Audio", use_container_width=True, key="tab_tts"):
        _run_tts(text_input, lang_code)


def _run_tts(text: str, lang: str):
    """Ejecuta el modelo TTS y muestra el reproductor de audio."""
    with st.spinner("Generando audio... puede tomar unos segundos"):
        result = text_to_speech(text, lang=lang)

    if result["status"] != "success":
        st.error(f"❌ Error: {result.get('error')}")
        return

    st.divider()
    st.markdown("#### 🎧 Resultado:")

    # Reproductor de audio
    st.audio(result["audio_bytes"], format="audio/wav")

    # Metricas
    st.divider()
    st.markdown("#### 📊 Métricas:")

    m1, m2, m3 = st.columns(3)
    m1.metric("Tiempo (s)", f"{result['time_seconds']}s")
    m2.metric("Caracteres", result["text_length"])
    m3.metric("Modelo", result["model"].split("/")[-1])
