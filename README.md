# 🤖 Transformer Web App

**Una aplicación web moderna para Traducción, Texto a Audio e Imagen usando Transformers.**

## 🎯 Características

### 1. **💬 Chat Interface**
- Chat en tiempo real con IA
- Tres acciones directas:
  - 🔊 **Escuchar** — Texto a audio con MMS-TTS
  - 🌐 **Traducir a ES** — Inglés → Español
  - 🌐 **Traducir a EN** — Español → Inglés
- Historial de conversación

### 2. **🖼️ Procesamiento de Imágenes**
- Sube imágenes (JPG, PNG)
- OCR automático (Tesseract) — Extrae texto
- Luego escucha o traduce el texto extraído
- Indicador de confianza OCR

### 3. **🔊 Texto a Audio**
- Convierte texto a voz (inglés y español)
- Modelos: MMS-TTS (Facebook)
- Reproductor de audio integrado

### 4. **🌐 Traductor**
- English ↔ Spanish
- Modelos: MarianMT (Helsinki-NLP)
- Rápido y preciso

---

## 🚀 Instalación

### Requisitos Previos
- Python 3.10+
- Virtual environment activo

### 1. Instalar Dependencias

```bash
# Desde el directorio web_app
pip install -r requirements_web.txt
```

### 2. (Opcional) Instalar Tesseract para OCR

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Descarga desde: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Verificar Instalación

```bash
python -c "import streamlit; import transformers; print('✅ OK')"
```

---

## 💻 Uso

### Iniciar la aplicación

```bash
# Desde la carpeta del proyecto raíz
cd web_app
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`

### Navegación

**4 Pestañas principales:**

1. **💬 Chat** — Interfaz conversacional rápida
2. **🖼️ Imagen** — Sube imágenes, extrae texto con OCR
3. **🔊 Texto a Audio** — Convierte texto a voz
4. **🌐 Traducción** — Traductor dedicado

---

## 🔧 Configuración

### config.yaml

Parámetros ajustables:

```yaml
# Texto a Audio
text_to_speech:
  model_en: "facebook/mms-tts-eng"
  model_es: "facebook/mms-tts-spa"

# Traducción
translation:
  en_to_es: "Helsinki-NLP/opus-mt-en-es"
  es_to_en: "Helsinki-NLP/opus-mt-es-en"

# Imagen
ocr:
  language: "eng"
  max_width: 1024
```

---

## 📚 Arquitectura Técnica

```
web_app/
├── app.py                    [Main Streamlit app]
├── utils/
│   ├── __init__.py
│   ├── models.py            [Model loading + caching]
│   ├── translation.py       [MarianMT wrapper]
│   ├── text_to_speech.py    [MMS-TTS wrapper]
│   └── image_processor.py   [OCR + image handling]
├── requirements_web.txt
└── README.md
```

### Modelos Utilizados

| Tarea | Modelo | Tamaño | Velocidad |
|-------|--------|--------|-----------|
| **Texto a Audio (EN)** | facebook/mms-tts-eng | ~250M | ⚡ Rápido |
| **Texto a Audio (ES)** | facebook/mms-tts-spa | ~250M | ⚡ Rápido |
| **Traducción EN→ES** | Helsinki-NLP/opus-mt-en-es | 298M | ⚡ Rápido |
| **Traducción ES→EN** | Helsinki-NLP/opus-mt-es-en | 298M | ⚡ Rápido |
| **OCR** | Tesseract | Local | ⚡ Muy rápido |

---

## 🎓 Cómo Funciona

### Texto a Audio (MMS-TTS)

1. Recibe texto en inglés o español
2. Procesa con modelo MMS-TTS de Facebook
3. Genera audio WAV a 16kHz
4. Reproduce en el navegador con st.audio()

### Traducción (MarianMT)

1. Carga modelo MarianMT para par de idiomas
2. Tokeniza texto origen
3. Codifica y decodifica a idioma destino
4. Devuelve traducción

### OCR (Tesseract)

1. Lee imagen con PIL
2. Aplica Tesseract OCR
3. Calcula confianza OCR
4. Retorna texto extraído

---

## ⚡ Optimizaciones

### Caching de Modelos
```python
@st.cache_resource
def load_translation_model(model_name):
    # Carga solo UNA vez
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer
```

**Beneficio:** Primera ejecución tarda ~10s, posteriores <100ms

### GPU Support
Automático con CUDA si está disponible. Fallback a CPU.

### Batch Processing
Internamente procesa lotes para mayor velocidad.

---

## 🐛 Troubleshooting

### "pytesseract not found"
```bash
pip install pytesseract
# Luego instala Tesseract en tu SO (ver Instalación)
```

### "Model not found" o "Connection timeout"
- Verifica conexión a internet (descarga modelos de HuggingFace)
- Los modelos se cachean en `~/.cache/huggingface/`

### "CUDA out of memory"
- Reduce max_length
- Usa CPU (automático fallback)
- Reinicia la app

### "Tesseract not installed"
- Sin Tesseract, OCR muestra un placeholder
- Instala Tesseract para OCR real (ver Instalación)

---

## 📊 Performance

### Tiempos Típicos (primera ejecución, GPU RTX 3090)

| Operación | Tiempo |
|-----------|--------|
| Cargar modelo | ~5-10s |
| Traducir (oración) | 50-100ms |
| Texto a Audio (oración corta) | 1-3s |
| OCR (imagen A4) | 500-1000ms |

*Sin GPU (CPU): 3-5x más lento*

---

## 🔐 Seguridad

- ✅ No almacena datos de usuarios (stateless)
- ✅ Todo local (no envía a servidores externos)
- ✅ Inputs validados
- ✅ Manejo robusto de errores

---

## 🚧 Futuro

Próximas features:

- [ ] Soporte para más idiomas
- [ ] Detección de idioma automática
- [ ] Análisis de sentimiento
- [ ] Descarga de resultados (PDF, TXT)
- [ ] Historial persistente (base de datos)
- [ ] API REST (FastAPI backend)
- [ ] Deploy en cloud (Hugging Face Spaces, Heroku)

---

## 📖 Ejemplos de Uso

### Ejemplo 1: Escuchar Texto

```
Input: "Hello, welcome to the Transformer Web App"
Idioma: Inglés
Acción: 🔊 Generar Audio
Output: [Audio WAV reproducible]
Tiempo: ~2s
```

### Ejemplo 2: Traducir con Imagen

```
Input: [Subir imagen con texto en inglés]
Acción 1: OCR (extrae texto)
Acción 2: 🌐 Traducir a ES
Output: [Texto en español]
Tiempo: ~1500ms (OCR) + 100ms (Traducción)
```

### Ejemplo 3: Chat Rápido

```
Usuario: "The weather is beautiful today"
Acción: 🔊 Escuchar
Output: [Audio generado]
Usuario: "Translate to Spanish"
Acción: 🌐 Traducir a ES
Output: [El clima está hermoso hoy]
```

---

## 🤝 Contribuir

¿Quieres mejorar la app?

1. Fork del repo
2. Crea rama: `git checkout -b feature/mi-feature`
3. Commit: `git commit -m "Add feature"`
4. Push: `git push origin feature/mi-feature`
5. Pull request

---

## 📝 Licencia

MIT License

---

## 📞 Soporte

- 📧 Email: support@transformerweb.dev
- 🐛 Issues: GitHub Issues
- 💬 Chat: Discord community

---

## 🙏 Agradecimientos

- 🤗 Hugging Face Transformers
- ⚡ PyTorch
- 🎈 Streamlit
- 🔤 Tesseract OCR
- 🇪🇸 Comunidad de traducción Helsinki-NLP

---

**Made with ❤️ using Transformers**

*Última actualización: 2026-03-30*
