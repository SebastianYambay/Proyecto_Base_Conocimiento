"""
Detección de objetos en vivo con YOLO (COCO) sobre la webcam vía WebRTC.
Autor: Sebastian Yambay
Repositorio: https://github.com/SebastianYambay/Proyecto_Base_Conocimiento

Cada frame de la cámara se procesa con YOLOv8: se dibujan las cajas de
detección y, si aparece alguna de las clases objetivo configuradas, se dispara
una alerta de Telegram (con enfriamiento para no saturar).

El modelo por defecto es yolov8n.pt (liviano, 80 clases COCO). Se puede
sobreescribir con la variable de entorno YOLO_MODEL.
"""

import os
import time
import threading

import av
from ultralytics import YOLO

# Reutiliza el mismo emisor de alertas (texto) que usa el dashboard.
from src.telegram_bot import send_telegram_alert

# Modelo configurable (por defecto yolov8n COCO, liviano)
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")

_model = None
_model_lock = threading.Lock()


def get_yolo_model():
    """Carga el modelo YOLO una sola vez (thread-safe, cacheado a nivel módulo)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(YOLO_MODEL_PATH)
    return _model


def get_class_names():
    """Lista ordenada de nombres de clases que el modelo puede detectar."""
    model = get_yolo_model()
    return list(model.names.values())


def detect_on_frame(img_bgr, conf=0.5):
    """
    Corre YOLO sobre un frame BGR (formato OpenCV) y devuelve:
      - annotated: ndarray BGR con las cajas dibujadas
      - detections: lista de (label, confianza)
    Se usa en el modo "Cámara IP" (lectura de stream con OpenCV).
    """
    model = get_yolo_model()
    result = model.predict(img_bgr, conf=conf, verbose=False)[0]
    annotated = result.plot()
    names = result.names
    detections = [(names[int(b.cls[0])], float(b.conf[0])) for b in result.boxes]
    return annotated, detections


class YOLOVideoProcessor:
    """
    Procesador de frames para streamlit-webrtc.

    streamlit-webrtc crea una instancia y llama a recv() por cada frame en un
    hilo aparte. Los parámetros (clases objetivo, confianza, chat_id) se
    inyectan desde el hilo principal de Streamlit en cada rerun.
    """

    def __init__(self):
        self.model = get_yolo_model()
        self.conf_threshold = 0.5
        self.target_classes = set()       # nombres de clases que disparan alerta
        self.chat_id = None               # destino Telegram (si está sincronizado)
        self.alert_cooldown = 30          # segundos mínimos entre alertas
        self._last_alert_ts = 0.0
        self._lock = threading.Lock()
        self.last_detections = []         # [(label, conf), ...] para la UI

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Inferencia YOLO sobre el frame actual
        result = self.model.predict(img, conf=self.conf_threshold, verbose=False)[0]
        annotated = result.plot()  # ndarray BGR con las cajas dibujadas

        names = result.names
        detected = []
        for box in result.boxes:
            label = names[int(box.cls[0])]
            conf = float(box.conf[0])
            detected.append((label, conf))

        with self._lock:
            self.last_detections = detected

        # ¿Aparece alguna clase objetivo? -> alerta de texto
        hits = sorted({lbl for lbl, _ in detected if lbl in self.target_classes})
        if hits:
            self._maybe_alert(hits)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def _maybe_alert(self, hits):
        """Envía la alerta de Telegram (texto) respetando el enfriamiento."""
        now = time.time()
        if now - self._last_alert_ts < self.alert_cooldown:
            return
        if not self.chat_id:
            return  # sin destino sincronizado, no se envía

        self._last_alert_ts = now
        labels = ", ".join(hits)
        msg = (
            "🚨 *ALERTA DE CÁMARA (YOLO)*\n\n"
            f"Objeto(s) detectado(s): {labels}\n"
            f"Hora: {time.strftime('%H:%M:%S')}"
        )

        chat_id = self.chat_id

        # Enviar en un hilo aparte para NO congelar el video
        def _send():
            try:
                send_telegram_alert(msg, chat_id)
            except Exception:
                pass

        threading.Thread(target=_send, daemon=True).start()

    def snapshot_detections(self):
        """Copia segura de las últimas detecciones para mostrar en la UI."""
        with self._lock:
            return list(self.last_detections)
