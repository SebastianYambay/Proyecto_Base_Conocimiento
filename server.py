"""
Transformer Web App - Servidor Web Flask con Nueva Interfaz Stitch
Proporciona la interfaz interactiva en Tailwind CSS y conecta los modelos Transformer.
"""

import os
import base64
import logging
from flask import Flask, request, jsonify, render_template

from utils.translation import translate
from utils.text_to_speech import text_to_speech
from utils.image_processor import extract_text_from_image
from utils.models import get_device_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)


# ==================== RUTAS DE VISTAS (UI) ====================

@app.route("/")
@app.route("/chat")
def page_chat():
    return render_template("chat.html")


@app.route("/ocr")
def page_ocr():
    return render_template("ocr.html")


@app.route("/tts")
def page_tts():
    return render_template("tts.html")


@app.route("/translator")
def page_translator():
    return render_template("translator.html")


# ==================== API ENDPOINTS ====================

@app.route("/api/translate", methods=["POST"])
def api_translate():
    """Traducción de texto inglés <-> español con MarianMT."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    source_lang = data.get("source_lang", "en")
    target_lang = data.get("target_lang", "es")
    logger.info(f"Petición de traducción: {source_lang} -> {target_lang}")
    result = translate(text, source_lang=source_lang, target_lang=target_lang)
    return jsonify(result)


@app.route("/api/tts", methods=["POST"])
def api_tts():
    """Síntesis de voz con Facebook MMS-TTS. Devuelve audio en base64 para reproducción inmediata."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    lang = data.get("lang", "es")
    logger.info(f"Petición TTS en idioma: {lang}")
    result = text_to_speech(text, lang=lang)
    if result.get("status") == "success" and result.get("audio_bytes"):
        audio_b64 = base64.b64encode(result["audio_bytes"]).decode("utf-8")
        return jsonify({
            "status": "success",
            "audio_base64": audio_b64,
            "sampling_rate": result.get("sampling_rate", 16000),
            "time_seconds": result.get("time_seconds", 0),
            "model": result.get("model", ""),
            "text": text
        })
    return jsonify(result)


@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    """Extracción de texto OCR de imagen con Tesseract."""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No se envió ningún archivo"})
    file = request.files["file"]
    image_bytes = file.read()
    logger.info(f"Petición OCR para archivo: {file.filename}")
    result = extract_text_from_image(image_bytes)
    return jsonify(result)


@app.route("/api/system", methods=["GET"])
def api_system():
    """Información del hardware y aceleración GPU."""
    return jsonify(get_device_info())


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Transformer Web App (Stitch UI) corriendo en:")
    print("-> http://localhost:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
