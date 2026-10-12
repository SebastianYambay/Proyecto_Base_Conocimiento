"""Componentes compartidos: estilos CSS, estado de sesion y mensajes del chat."""

import streamlit as st
from utils.models import get_device_info


def inject_custom_css():
    """Inyecta los estilos personalizados de la app."""
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .section-header {
            font-size: 1.3rem;
            font-weight: bold;
            margin-top: 1rem;
            padding: 0.5rem;
            background: linear-gradient(90deg, #ff6b6b, #ffd93d);
            border-radius: 0.5rem;
            color: white;
        }
        .success-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .error-box {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .metric-box {
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Inicializa el estado de sesion de Streamlit (mensajes + info del dispositivo)."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "device_info" not in st.session_state:
        st.session_state.device_info = get_device_info()


def add_message(role: str, content: str, metadata: dict = None):
    """Agrega un mensaje al historial del chat."""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "metadata": metadata or {}
    })


def display_message(role: str, content: str, metadata: dict = None):
    """Muestra un mensaje en la interfaz del chat."""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)
            if metadata:
                if "time_seconds" in metadata:
                    st.caption(f"⏱️ Tiempo: {metadata['time_seconds']}s")
