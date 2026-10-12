"""Pestana Traduccion: ingles <-> espanol con MarianMT."""

import streamlit as st
from utils.translation import translate

# Mapa de idiomas para el selectbox
LANG_MAP = {
    "Inglés (en)": "en",
    "Español (es)": "es",
}


def render_translation_tab():
    """Interfaz dedicada para traduccion."""
    st.markdown("### 🌐 Traductor")
    st.markdown("Traduce entre inglés y español (MarianMT)")

    col_lang1, col_lang2 = st.columns(2)
    with col_lang1:
        source_lang = st.selectbox("De:", list(LANG_MAP.keys()), key="source_lang")
    with col_lang2:
        target_lang = st.selectbox("A:", list(LANG_MAP.keys()), index=1, key="target_lang")

    source_code = LANG_MAP[source_lang]
    target_code = LANG_MAP[target_lang]

    # No tiene sentido traducir al mismo idioma
    if source_code == target_code:
        st.warning("⚠️ Selecciona idiomas diferentes")
        return

    text_input = st.text_area(
        "Texto a traducir:",
        placeholder="Escribe el texto a traducir...",
        height=200,
        key="translate_input"
    )

    if not text_input:
        return

    if st.button("🚀 Traducir", use_container_width=True, key="tab_translate"):
        _run_translation(text_input, source_code, target_code, source_lang, target_lang)


def _run_translation(text: str, source_code: str, target_code: str,
                     source_label: str, target_label: str):
    """Ejecuta la traduccion y muestra el resultado."""
    with st.spinner("Traduciendo..."):
        result = translate(text, source_code, target_code)

    if result["status"] != "success":
        st.error(f"❌ Error: {result.get('error')}")
        return

    st.divider()
    st.markdown("#### 🌐 Resultado:")

    col_orig, col_trans = st.columns(2)

    with col_orig:
        st.markdown(f"**{source_label}:**")
        st.text_area("Original", value=text, height=200, disabled=True, label_visibility="collapsed")

    with col_trans:
        st.markdown(f"**{target_label}:**")
        st.success(result["translated"])

    st.divider()
    st.metric("⏱️ Tiempo", f"{result['time_seconds']}s")
