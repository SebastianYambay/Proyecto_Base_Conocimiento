"""
Transformer Web App
Traduccion, texto a audio y procesamiento de imagenes con modelos Transformer.
"""

import streamlit as st
import logging

from components.common import inject_custom_css, initialize_session_state
from components.sidebar import render_sidebar
from components.chat_tab import render_chat_tab
from components.image_tab import render_image_tab
from components.summary_tab import render_tts_tab
from components.translation_tab import render_translation_tab

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuracion de la pagina
st.set_page_config(
    page_title="🤖 Transformer Web App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Punto de entrada. Arma la interfaz con tabs."""
    inject_custom_css()
    initialize_session_state()

    st.markdown("""
        <h1 class="main-header">🤖 Transformer Web App</h1>
        <p><strong>Traducción, Texto a Audio e Imagen</strong></p>
        <hr>
    """, unsafe_allow_html=True)

    render_sidebar()

    # Pestanas principales
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Chat", "🖼️ Imagen", "🔊 Texto a Audio", "🌐 Traducción"]
    )

    with tab1:
        render_chat_tab()
    with tab2:
        render_image_tab()
    with tab3:
        render_tts_tab()
    with tab4:
        render_translation_tab()


if __name__ == "__main__":
    main()
