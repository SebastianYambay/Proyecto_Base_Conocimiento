# ⚡ Quick Start — Transformer Web App

## 🚀 En 5 Minutos

### 1️⃣ Instalar

```bash
# Desde web_app/
pip install -r requirements_web.txt
```

**Con setup script (recomendado):**
```bash
bash setup.sh
```

### 2️⃣ Correr la App

```bash
streamlit run app.py
```

Abrirá automáticamente en: `http://localhost:8501`

### 3️⃣ ¡Usar!

Tienes **4 opciones**:

#### 💬 **Chat** (Lo más rápido)
1. Escribe texto
2. Presiona: 🔊 Escuchar / 🌐 Traducir ES / 🌐 Traducir EN
3. Listo ✅

#### 🖼️ **Imagen**
1. Sube JPG/PNG
2. Automáticamente extrae texto (OCR)
3. Escucha o traduce el texto

#### 🔊 **Texto a Audio**
1. Elige idioma (inglés o español)
2. Escribe o pega texto
3. Genera audio y escucha en el navegador

#### 🌐 **Traductor**
1. Elige idiomas (EN ↔ ES)
2. Pega texto
3. Traduce

---

## 📦 Requisitos Mínimos

```bash
Python 3.10+
pip
internet (para descargar modelos)
```

### GPU (Opcional pero recomendado)

Con GPU: ~100ms por operación
Sin GPU: ~300-500ms por operación

---

## ⚙️ Configuración Rápida

### Solo CPU
```python
# En app.py, línea ~15
DEVICE = "cpu"  # Fuerza CPU
```

### Texto a Audio
```python
# Los modelos por idioma están en utils/text_to_speech.py
TTS_MODELS = {
    "en": "facebook/mms-tts-eng",
    "es": "facebook/mms-tts-spa",
}
```

---

## 🔧 Troubleshooting Rápido

| Problema | Solución |
|----------|----------|
| "Module not found" | `pip install -r requirements_web.txt` |
| "CUDA out of memory" | Reduce max_length o usa CPU |
| "Model download failed" | Espera, verifica internet |
| "Tesseract not found" | Opcional, OCR usa placeholder |

---

## 📊 Ejemplos Rápidos

### Escuchar Texto
```
Input: "Hello, welcome to the app"
Click: 🔊 Generar Audio
Output: Audio WAV reproducible
Time: ~2s
```

### Traducir Tweet
```
Input: "Great news about AI"
Click: 🌐 Traducir a ES
Output: "Grandes noticias sobre IA"
Time: ~100ms
```

### Leer Imagen
```
Upload: Foto con texto en inglés
OCR: Extrae automáticamente
Click: 🌐 Traducir a ES
Output: Texto en español
Time: ~1000ms
```

---

## 🎯 Casos de Uso

✅ **Accesibilidad**: Texto a voz para contenido
✅ **Traducción**: Documentos al español
✅ **OCR**: Extraer texto de imágenes
✅ **Educación**: Entender cómo funcionan Transformers
✅ **Prototipado**: Testing de modelos

---

## 📚 Recursos

- **App completa**: `app.py` (600+ líneas comentadas)
- **Utilidades**: `utils/` (4 módulos especializados)
- **Documentación**: `README.md` (detallado)
- **Modelos**: Automáticamente descargados de HuggingFace

---

## 🔌 Integración con el Proyecto Principal

Esta web app usa **exactamente lo que aprendiste**:

- **Atención**: Visualizada en traducción
- **Traducción**: MarianMT (Módulo 2)
- **Texto a Audio**: MMS-TTS (Facebook)
- **BERT/GPT**: Mencionados en análisis

---

## 💡 Tips

1. **Primera ejecución tarda**: Los modelos se descargan (~5GB), espera
2. **Luego es rápido**: Cacheados localmente, <100ms
3. **GPU 10x más rápido**: Si tienes CUDA, úsalo
4. **Prueba con ejemplos cortos**: Mientras aprendes

---

## 🚀 Siguiente

Después de explorar:
1. Modifica `utils/` para tus propios modelos
2. Agrega más idiomas en traducción
3. Deploy con Streamlit Cloud o Heroku

---

**¡Que disfrutes!** 🎉

Cualquier duda, revisa `README.md` para documentación completa.
