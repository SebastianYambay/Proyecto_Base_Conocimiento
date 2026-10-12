"""Pestana Chat: escuchar texto (TTS) y traducir."""

import streamlit as st
from components.common import add_message, display_message
from utils.text_to_speech import text_to_speech
from utils.translation import translate


def render_chat_tab():
    """Interfaz de chat con botones de accion."""
    st.markdown("### 💬 Chat con Transformers")
    st.markdown("Escribe texto, luego elige la acción.")

    col1, col2 = st.columns([3, 1])

    with col1:
        user_input = st.text_area(
            "Escribe aquí:",
            placeholder="Escribe texto en inglés o español...",
            height=100,
            key="chat_input"
        )

    # Historial de mensajes
    st.divider()
    st.markdown("#### Historial de Chat")

    if st.session_state.messages:
        for msg in st.session_state.messages:
            display_message(msg["role"], msg["content"], msg.get("metadata"))
    else:
        st.info("👋 Inicia una conversación escribiendo algo")

    # Botones de accion
    if user_input:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button("🔊 Escuchar", use_container_width=True, key="chat_tts"):
                _handle_tts(user_input)
        with col_b:
            if st.button("🌐 Traducir a ES", use_container_width=True, key="chat_translate_es"):
                _handle_translate(user_input, "en", "es", "Español")
        with col_c:
            if st.button("🌐 Traducir a EN", use_container_width=True, key="chat_translate_en"):
                _handle_translate(user_input, "es", "en", "Inglés")


def _handle_tts(text: str):
    """Genera audio a partir del texto ingresado."""
    with st.spinner("Generando audio..."):
        add_message("user", f"**Escuchar:** {text[:100]}...")
        result = text_to_speech(text, lang="en")

        if result["status"] == "success":
            st.audio(result["audio_bytes"], format="audio/wav")
            add_message("assistant", "🔊 Audio generado", result)
            st.success("✅ Completado")
        else:
            st.error(f"❌ Error: {result.get('error')}")


def _handle_translate(text: str, source: str, target: str, label: str):
    """Traduce el texto y lo muestra en el chat."""
    with st.spinner("Traduciendo..."):
        add_message("user", f"**Traducir a {label}:** {text[:100]}...")
        result = translate(text, source_lang=source, target_lang=target)

        if result["status"] == "success":
            display_message("assistant", f"**{label}:** {result['translated']}", result)
            add_message("assistant", result["translated"], result)
            st.success("✅ Completado")
        else:
            st.error(f"❌ Error: {result.get('error')}")
