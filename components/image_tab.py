"""Pestana Imagen: subir imagen, extraer texto con OCR, luego escuchar o traducir."""

import streamlit as st
from PIL import Image

from utils.image_processor import extract_text_from_image, validate_image, resize_image
from utils.text_to_speech import text_to_speech
from utils.translation import translate


def render_image_tab():
    """Permite subir una imagen y procesarla con OCR."""
    st.markdown("### 🖼️ Procesar Imágenes")
    st.markdown("Sube una imagen en inglés. Extraeremos el texto y podrás escucharlo o traducirlo.")

    uploaded_image = st.file_uploader(
        "Sube una imagen (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        image = resize_image(image)

        col_img, col_opts = st.columns([2, 3])

        with col_img:
            st.image(image, caption="Imagen cargada")

        with col_opts:
            _process_image(image)


def _process_image(image: Image.Image):
    """Valida la imagen y extrae texto con OCR."""
    validation = validate_image(image)

    if validation["status"] != "valid":
        st.error(f"❌ Imagen inválida: {validation.get('error')}")
        return

    st.success(f"""
    ✅ Imagen válida
    - Formato: {validation['format']}
    - Tamaño: {validation['size']}
    """)

    # Extraer texto con Tesseract
    with st.spinner("Extrayendo texto (OCR)..."):
        ocr_result = extract_text_from_image(image)

    if ocr_result["status"] not in ["success", "warning"]:
        st.error(f"❌ Error en OCR: {ocr_result.get('error')}")
        return

    _display_extracted_text(ocr_result)


def _display_extracted_text(ocr_result: dict):
    """Muestra el texto extraido y los botones para escuchar o traducir."""
    st.markdown("#### Texto extraído:")
    extracted_text = ocr_result["text"]

    st.text_area("Texto:", value=extracted_text, height=150, disabled=True, key="ocr_text")

    if ocr_result["confidence"] > 0:
        st.caption(f"📊 Confianza OCR: {ocr_result['confidence']:.1f}%")

    st.divider()
    st.markdown("#### Acciones:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔊 Escuchar", use_container_width=True, key="img_tts"):
            with st.spinner("Generando audio..."):
                tts_result = text_to_speech(extracted_text, lang="en")
                if tts_result["status"] == "success":
                    st.markdown("**Audio:**")
                    st.audio(tts_result["audio_bytes"], format="audio/wav")
                    st.caption(f"Tiempo: {tts_result['time_seconds']}s")
                else:
                    st.error(tts_result.get("error"))

    with col2:
        if st.button("🌐 Traducir", use_container_width=True, key="img_translate"):
            with st.spinner("Traduciendo..."):
                trans_result = translate(extracted_text, source_lang="en", target_lang="es")
                if trans_result["status"] == "success":
                    st.markdown("**Español:**")
                    st.success(trans_result["translated"])
                else:
                    st.error(trans_result.get("error"))
