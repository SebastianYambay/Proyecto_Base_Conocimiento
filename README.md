# 💧 SIPCA — Sistema de Predicción de Calidad de Agua
**Machine Learning + Visión por Computadora + IA local para evaluar la potabilidad del agua**

- **Autor:** [Sebastian Yambay](https://github.com/SebastianYambay)
- **Repositorio:** [SebastianYambay/Proyecto_Base_Conocimiento](https://github.com/SebastianYambay/Proyecto_Base_Conocimiento)

---

---

## 📝 Descripción del Proyecto

**SIPCA** es un dashboard interactivo (Streamlit) que evalúa la **potabilidad del agua** combinando varias técnicas de IA:

- **Predicción con Machine Learning:** un modelo *RandomForest* clasifica una muestra como **Potable / No Potable** a partir de 9 parámetros físico-químicos (pH, dureza, sólidos, cloraminas, sulfatos, conductividad, carbono orgánico, trihalometanos y turbidez).
- **Análisis de imágenes (turbidez):** un modelo de **visión local** estima la turbidez (NTU) a partir de una foto del agua.
- **Detección en vivo con YOLO:** usa la **webcam o la cámara del celular** para detectar objetos/contaminantes en tiempo real y disparar alertas.
- **Alertas por Telegram:** notificaciones automáticas cuando el agua es no potable o cuando una cámara detecta un objeto objetivo.
- **Asistente IA (chatbot):** responde consultas sobre calidad de agua usando un **LLM local** (sin depender de APIs de pago).

> 🔒 **100% local por defecto:** el chatbot y el análisis de imágenes usan modelos que corren en tu propia máquina vía **LM Studio** (o cualquier servidor compatible con la API de OpenAI). No se envía nada a OpenAI ni a Google.

---

## 📸 Imágenes del Dashboard

| Dashboard Principal | Predicción y Alertas |
|---------------------|----------------------|
| ![Dashboard](/assets/img/dashboard_general.png) | ![Prediccion](/assets/img/prediccion.png) |

| Asistente IA | Análisis de Visión |
|--------------|--------------------|
| ![Chatbot](/assets/img/asistente_ia.png) | ![Vision](/assets/img/vision_img.png) |

---

## ⚙️ Requisitos previos

- **Python 3.10+**
- **(Opcional) LM Studio** — solo si usarás el chatbot y/o el análisis de imágenes.
  Descárgalo en [lmstudio.ai](https://lmstudio.ai).
- **(Opcional) Cuenta de Telegram** — solo si quieres recibir alertas.

> El modelo de predicción **ya viene entrenado** en `models/`. **No necesitas correr los notebooks** para usar la app; esos solo sirven si quieres reentrenar el modelo desde cero.

---

## 🚀 Ejecución paso a paso (Local)

### 1. Clonar e instalar dependencias
```powershell
git clone https://github.com/SebastianYambay/Proyecto_Base_Conocimiento.git
cd Proyecto_Base_Conocimiento

# Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS / Linux

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar variables de entorno
Copia `.env.example` a `.env` y complétalo:
```powershell
copy .env.example .env
```
```env
# --- LLM LOCAL (por defecto, sin APIs externas) ---
LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1
LOCAL_LLM_MODEL=llama-2-7b-chat          # modelo del chatbot (solo texto)
LOCAL_VISION_MODEL=qwen2-vl-2b-instruct  # modelo de visión (turbidez)
LOCAL_LLM_API_KEY=lm-studio

# --- TELEGRAM (para alertas) ---
TELEGRAM_TOKEN=tu_token_de_@BotFather
```

### 3. (Opcional) Preparar LM Studio para chatbot/visión
1. Abre **LM Studio** → pestaña **Search** y descarga:
   - Un modelo de **texto** para el chat (ej. `llama-2-7b-chat`).
   - Un modelo **multimodal/visión** para la turbidez (ej. `qwen2-vl-2b-instruct`, con insignia 👁️).
2. **Carga** ambos modelos (*Load Model*).
3. Activa el servidor (pestaña **Developer** → *Status: Running*), debe quedar en `http://127.0.0.1:1234`.
4. Copia el **API Model Identifier** de cada modelo y ponlos en `LOCAL_LLM_MODEL` / `LOCAL_VISION_MODEL` del `.env`.

### 4. Ejecutar la aplicación
```powershell
streamlit run app.py
```
Abre **http://localhost:8501**. El bot de Telegram se inicia automáticamente en segundo plano.

---

## 🐳 Ejecución con Docker (alternativa)

```powershell
docker compose up --build      # app en http://localhost:8501
```
- **LM Studio debe correr en el HOST** (no en el contenedor); el `docker-compose.yml` ya apunta el LLM a `host.docker.internal:1234`.
- La primera build es grande (incluye PyTorch + YOLO) y pre-descarga `yolov8n.pt`.

---

## 🤖 Guía de Uso

### Predicción de potabilidad (Dashboard General)
1. Ajusta los parámetros de la muestra con los sliders de la barra lateral.
2. Pulsa **«Analizar Muestra»** → verás el resultado (Potable/No Potable) con % de confianza y gráficos.
3. **Análisis por lotes:** sube un CSV (una fila por muestra) para predecir muchas a la vez.

### Alertas por Telegram
1. En Telegram, busca **tu bot** (el que creaste con @BotFather) y envía **`/start`**.
2. En el Dashboard → **🔔 Conectar Alertas** → **«Sincronizar con Bot»** (debe decir "Conectado").
3. Recibirás alertas cuando el agua sea **No Potable**, el **pH** esté fuera de norma, o una cámara detecte un objeto objetivo.

### Análisis de Imágenes (turbidez)
- Sube una foto del agua → **«Analizar Turbidez»** → estima el NTU y su clasificación. Requiere un modelo de visión cargado en LM Studio.

### Revisión en Vivo (YOLO)
- Elige la fuente: **Webcam del navegador** o **Cámara IP (celular)**.
  - *Cámara IP:* instala una app tipo **IP Webcam** en el celular (misma WiFi) y pega su URL `http://<ip-del-cel>:8080/video`.
- Selecciona qué objetos disparan la alerta (ej. `bottle`, `cup`, `person`).
- Al detectar un objetivo, se envía una alerta de texto a Telegram (respeta un enfriamiento configurable).

### Asistente IA
- Botón flotante 🤖 (abajo a la derecha) → barra lateral **«Configurar Asistente IA»** → **Conectar**. Usa el LLM local; no necesita API key.

---

## 📂 Estructura del Proyecto
```
Proyecto_Base_Conocimiento/
├── app.py                      # Aplicación principal (Streamlit)
├── models/                     # Modelo y scaler ya entrenados (.pkl)
├── notebooks/                  # EDA, limpieza y entrenamiento (solo para reentrenar)
├── cameras/                    # Imágenes y metadata del monitoreo de cámaras
├── src/
│   ├── chatbot_llm.py          # Chatbot (LLM local + proveedores nube opcionales)
│   ├── vision_module.py        # Análisis de turbidez por imagen (visión local)
│   ├── camera_detection.py     # Detección YOLO en vivo (webcam WebRTC)
│   ├── telegram_bot.py         # Bot de Telegram (alertas + comandos)
│   ├── preprocessing.py        # Pipeline de preprocesamiento
│   └── model_train.py          # Entrenamiento del modelo
├── requirements.txt
├── Dockerfile / docker-compose.yml
├── LICENSE
└── .env.example
```

---

## 🔁 (Opcional) Reentrenar el modelo
Solo si quieres regenerar el modelo (requiere los datos de los notebooks):
```powershell
cd src
python model_train.py          # debe ejecutarse DENTRO de src/
```

---

## 🛠️ Tecnologías
- **Core:** Python 3.10+, Streamlit
- **ML/Data:** Scikit-learn, XGBoost, Pandas, NumPy
- **Visión / Detección:** Ultralytics YOLOv8, OpenCV, streamlit-webrtc
- **IA Generativa (local):** LM Studio / Ollama vía API compatible con OpenAI
- **Notificaciones:** python-telegram-bot
- **Visualización:** Plotly

---

## 👤 Autor y Créditos

- **Autor:** [Sebastian Yambay](https://github.com/SebastianYambay)
- **Repositorio:** [https://github.com/SebastianYambay/Proyecto_Base_Conocimiento](https://github.com/SebastianYambay/Proyecto_Base_Conocimiento)
- **Licencia:** Distribuido bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

