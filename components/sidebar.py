"""Sidebar: info del sistema, modelos usados y controles."""

import streamlit as st


def render_sidebar():
    """Dibuja el sidebar con info del sistema y botones."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")

        # Info del dispositivo (CPU o GPU)
        st.subheader("Información del Sistema")
        device_info = st.session_state.device_info
        st.info(f"""
        **Device:** {device_info['device'].upper()}
        **CUDA Available:** {device_info['cuda_available']}
        """)

        if device_info['cuda_available'] == 'True':
            st.success(f"GPU disponible: {device_info.get('cuda_device', 'N/A')}")
            st.caption(f"Memoria: {device_info.get('cuda_memory', 'N/A')}")

        st.divider()

        # Boton para limpiar el historial del chat
        if st.button("🗑️ Limpiar historial"):
            st.session_state.messages = []
            st.success("Historial limpiado")

        st.divider()

        st.subheader("Modelos Utilizados")
        st.markdown("""
        - **Traducción:** MarianMT (Helsinki-NLP/opus-mt)
        - **Texto a Audio:** MMS-TTS (Facebook)
        - **OCR:** Tesseract (opcional)
        """)

        st.divider()
        st.markdown("""
        <small>Powered by 🤗 Transformers, PyTorch y Streamlit</small>
        """, unsafe_allow_html=True)
