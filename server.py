"""
Transformer Web App - Servidor Web Flask con Nueva Interfaz Stitch
Proporciona la interfaz interactiva en Tailwind CSS y conecta los modelos Transformer.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import base64
import logging
import threading
import time
from flask import Flask, request, jsonify, render_template

from utils.translation import translate, preload_translation_models
from utils.text_to_speech import text_to_speech, preload_tts_models
from utils.image_processor import extract_text_from_image, get_easyocr_reader
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

def generate_assistant_response(prompt: str) -> str:
    """Genera respuestas educativas inteligentes para el chat multimodal."""
    p = prompt.lower().strip()
    if not p:
        return "Escribe una pregunta para comenzar."

    if "fotosíntesis" in p or "fotosintesis" in p:
        return "La fotosíntesis es el proceso mediante el cual las plantas capturan la luz solar, agua y dióxido de carbono para transformarlos en oxígeno y nutrientes."
    elif "ciclo del agua" in p:
        return "El ciclo del agua describe la circulación continua del agua en la Tierra: evaporación hacia la atmósfera, condensación en nubes y precipitación hacia la superficie."
    elif "cuento" in p or "historia" in p:
        return "En un vasto universo digital, una pequeña red Transformer descubrió que mirando a todas las palabras al mismo tiempo podía comprender cualquier historia del mundo."
    elif "transformer" in p:
        return "La arquitectura Transformer utiliza mecanismos de auto-atención para procesar secuencias en paralelo, capturando relaciones complejas entre palabras sin bucles recurrentes."
    elif "inteligencia artificial" in p or " ia " in p or "redes neuronales" in p:
        return "La Inteligencia Artificial permite a los sistemas computacionales emular capacidades cognitivas como traducción de idiomas, reconocimiento de imágenes y síntesis de voz."
    elif "mms" in p or "voz" in p or "audio" in p or "tts" in p:
        return "El modelo Meta MMS-TTS transforma representaciones fonéticas en ondas acústicas continuas de 16 kHz con entonación natural y expresiva."
    elif "marian" in p or "traduc" in p:
        return "MarianMT es un Transformer Seq2Seq optimizado para traducción automática bidireccional de alta fidelidad entre inglés y español."
    elif "ocr" in p or "imagen" in p or "foto" in p or "documento" in p:
        return "El sistema OCR reconoce las letras y palabras contenidas en una imagen, convirtiéndolas en texto editable para traducción o voz."
    elif "hola" in p or "buenos días" in p or "buenas" in p:
        return "¡Hola! Bienvenido a Transformer Studio. Puedes hacerme preguntas, traducir contenido o convertir cualquier texto a audio."
    else:
        return f"Sobre «{prompt}»: Este tema puede analizarse mediante redes neuronales Transformer, traduciéndolo o sintetizándolo en voz con los controles inferiores."


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Genera la respuesta del asistente multimodal en el chat."""
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt", "")
    start_time = time.time()
    response_text = generate_assistant_response(prompt)
    elapsed = round(time.time() - start_time, 2)
    return jsonify({
        "status": "success",
        "response": response_text,
        "time_seconds": elapsed
    })


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
    """Extracción de texto OCR de imagen."""
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


def background_warmup():
    """Precarga modelos en memoria de forma silenciosa para que la interfaz responda al instante."""
    logger.info("Iniciando precarga de modelos en segundo plano...")
    try:
        preload_translation_models()
        preload_tts_models()
        get_easyocr_reader()
        logger.info("¡Precarga de modelos completada con éxito!")
    except Exception as e:
        logger.warning(f"Precarga en segundo plano: {e}")


if __name__ == "__main__":
    # Iniciar precarga en segundo plano
    threading.Thread(target=background_warmup, daemon=True).start()

    print("\n" + "="*50)
    print("Transformer Web App (Stitch UI) corriendo en:")
    print("-> http://localhost:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
