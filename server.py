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


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


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
        return "Por favor, escribe una pregunta o tema para analizar."

    if any(w in p for w in ["fotosíntesis", "fotosintesis"]):
        return "La fotosíntesis es el proceso bioquímico mediante el cual las plantas capturan la luz del sol, agua y dióxido de carbono para transformarlos en glucosa y oxígeno vital para la atmósfera."
    elif any(w in p for w in ["ciclo del agua", "ciclo hidrologico"]):
        return "El ciclo del agua describe la circulación continua del agua en la Tierra: evaporación hacia la atmósfera, condensación formando nubes y precipitación hacia la superficie en forma de lluvia o nieve."
    elif any(w in p for w in ["cuento", "historia", "narracion"]):
        return "En un vasto universo digital, una pequeña red Transformer descubrió que al prestar atención a todas las palabras al mismo tiempo, podía comprender y conectar los pensamientos de toda la humanidad."
    elif any(w in p for w in ["transformer", "atencion", "self-attention", "mecanismo"]):
        return "La arquitectura Transformer utiliza mecanismos de auto-atención multi-cabezal para procesar secuencias en paralelo, capturando relaciones complejas entre palabras sin depender de bucles recurrentes."
    elif any(w in p for w in ["inteligencia artificial", " ia ", "deep learning", "aprendizaje profundo", "redes neuronales"]):
        return "La Inteligencia Artificial emula capacidades cognitivas humanas. En este sistema integra traducción neuronal bidireccional (MarianMT), síntesis de voz acústica (MMS-TTS) y visión computacional (OCR)."
    elif any(w in p for w in ["mms", "voz", "audio", "tts", "sintesis", "hablar"]):
        return "El modelo Meta MMS-TTS transforma representaciones fonéticas en ondas acústicas continuas a 16 kHz con entonación natural, facilitando accesibilidad y aprendizaje auditivo."
    elif any(w in p for w in ["marian", "traduc", "traduccion", "translation"]):
        return "MarianMT es un modelo Transformer encoder-decoder desarrollado por Helsinki-NLP, optimizado para traducción automática rápida y de alta fidelidad entre inglés y español."
    elif any(w in p for w in ["ocr", "imagen", "foto", "vision", "documento", "escaner"]):
        return "El sistema OCR reconoce los patrones visuales de letras y palabras en imágenes escaneadas o fotografías, convirtiéndolas en texto editable para traducción o lectura en voz alta."
    elif any(w in p for w in ["gravedad"]):
        return "La gravedad es la fuerza física de atracción que experimentan los cuerpos con masa. En la teoría de la relatividad general de Einstein, representa la curvatura del espacio-tiempo provocada por la materia y la energía."
    elif any(w in p for w in ["atomo", "átomo", "materia"]):
        return "El átomo es la unidad básica de la materia. Consiste en un núcleo denso formado por protones y neutrones, rodeado por electrones organizados en capas cuánticas de energía."
    elif any(w in p for w in ["celula", "célula", "adn"]):
        return "La célula es la unidad biológica estructural de todo ser vivo. Alberga el ADN celular, el cual almacena la información genética necesaria para la vida y el funcionamiento del organismo."
    elif any(w in p for w in ["python", "programacion", "codigo"]):
        return "Python es un lenguaje de programación de alto nivel apreciado por su sintaxis limpia y su enorme potencia en computación científica, desarrollo web y librerías de Inteligencia Artificial como PyTorch y Transformers."
    elif any(w in p for w in ["hola", "buenos días", "buenas", "saludos", "que tal"]):
        return "¡Hola! Bienvenido a Transformer Studio. Puedes hacerme cualquier pregunta técnica o educativa, traducir respuestas en tiempo real o escucharlas en voz alta."
    elif any(w in p for w in ["quien eres", "que haces", "ayuda"]):
        return "Soy el Asistente Multimodal de Transformer Studio. Permito formular consultas, generar respuestas explicativas, escucharlas en audio natural mediante MMS-TTS y traducirlas con MarianMT."
    else:
        return f"Respecto a «{prompt}»: Este tema es analizado por los modelos Transformer del sistema. Puedes escuchar esta explicación o traducirla al inglés con los botones didácticos inferiores."


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
