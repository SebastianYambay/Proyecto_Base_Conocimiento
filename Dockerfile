# ---------------------------------------------------------
# SIPCA - Sistema de Predicción de Calidad de Agua
# Imagen de la app Streamlit (incluye el bot de Telegram en un hilo)
# ---------------------------------------------------------
FROM python:3.11-slim

# Evita prompts de apt y mejora logs de Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Dependencias de sistema mínimas para opencv-python (libGL/glib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias primero para aprovechar la caché de capas
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-descargar los pesos de YOLO para que la detección funcione offline en runtime
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copiar el resto del proyecto (ver .dockerignore para exclusiones)
COPY . .

# Configuración de Streamlit para correr en contenedor
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Healthcheck del endpoint interno de Streamlit
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)" || exit 1

CMD ["streamlit", "run", "app.py"]
