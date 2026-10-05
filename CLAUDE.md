# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SIPCA — Sistema de Predicción de Calidad de Agua. A Streamlit dashboard that classifies water samples as potable / non-potable using a RandomForest model, with Telegram alerting, an OpenAI/Gemini chatbot, and an OpenCV-based turbidity (NTU) vision module. Built by Sebastian Yambay (https://github.com/SebastianYambay/Proyecto_Base_Conocimiento) for Samsung Innovation Campus Ecuador 2025. Codebase is Spanish-language (UI strings, comments, identifiers); preserve Spanish when editing user-facing text.

Repository: `https://github.com/SebastianYambay/Proyecto_Base_Conocimiento`
Folder name: `Proyecto_Base_Conocimiento` (quote paths if they contain spaces).

## Commands

```powershell
# Install (Python 3.10+)
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Run the app (also boots the Telegram bot listener in a background thread)
streamlit run app.py

# Run via Docker (LM Studio/Ollama must run on the HOST, not in the container)
docker compose up --build      # app on http://localhost:8501

# Retrain the model — must be run from inside src/ because model_train.py uses
# `import preprocessing as prep` (sibling import, not a package import)
cd src
python model_train.py

# Standalone vision-module sanity check
python test_vision.py
```

Required `.env` at repo root (see `.env.example`):
- `TELEGRAM_TOKEN` — from @BotFather (bot won't start without it)
- `LOCAL_LLM_BASE_URL` / `LOCAL_LLM_MODEL` — local OpenAI-compatible server (LM Studio default `http://127.0.0.1:1234/v1`, model `mistral-7b-instruct-v0.1`). This is the **default** for both the chatbot and vision module; no cloud key needed.
- `LOCAL_VISION_MODEL` — must be a *multimodal* model for the turbidity image analysis (mistral-7b is text-only; load llava/qwen2-vl/moondream).
- Cloud keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) are **optional** — only needed if the user picks a cloud provider in the chatbot UI.

There is no test suite, linter config, or CI. `test_vision.py` is a manual script, not pytest.

## Architecture

**Entry point — `app.py`:** Streamlit single-page app. On import it calls `iniciar_bot_en_background()` (wrapped in `@st.cache_resource` so it runs exactly once per Streamlit process) which spawns a daemon thread running `telegram_bot.run_listener` — the bot lives in the same process as the dashboard.

**Cross-process state via JSON files at repo root** (this is the integration contract between modules — don't replace with in-memory state without updating all readers):
- `telegram_connection.json` — chat_id ↔ user mapping, written when a user `/start`s the bot, read by the dashboard's "Sincronizar con Telegram" button.
- `water_status.json` — latest prediction the dashboard wrote; the bot reads it to answer user queries.
- `maintenance_log.json` — reports submitted via Telegram, read by the dashboard.

**ML pipeline (`src/`):**
- `preprocessing.py` — `load_data`, `split_data`, `train_save_scaler`, `scale_data`. The scaler is fit on train and persisted to `models/scaler.pkl`; the app loads it at inference.
- `model_train.py` — trains RandomForest on `data/processed/water_potability_cleaned.csv` and writes `models/water_potability_model.pkl`. The `data/` directory is **not committed** — see `notebooks/01_eda_analisis.ipynb` → `02_limpieza_etl.ipynb` → `03_entrenamiento.ipynb` for the data origin/cleaning flow before retraining.
- Features (order matters for the scaler): pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity.

**Vision (`src/vision_module.py`):** sends the uploaded image to a vision LLM via an OpenAI-compatible endpoint (`LOCAL_LLM_BASE_URL`, model `LOCAL_VISION_MODEL`) and parses a strict-JSON turbidity estimate (NTU). `get_ntu_interpretation` maps the value to a human-readable category. Despite the README wording, this is **not** classical OpenCV processing — it requires a multimodal model. Sample images for the "Monitoreo de cámaras" view live in `cameras/` (metadata in `cameras/info.json`).

**Chatbot (`src/chatbot_llm.py`):** `create_chatbot_widget` renders a Streamlit chat UI. Default provider is `"local"` (LM Studio/Ollama via the OpenAI SDK pointed at `LOCAL_LLM_BASE_URL`); OpenAI/Gemini/Anthropic/OpenRouter remain as optional cloud fallbacks selectable in the sidebar. The local provider needs no API key.

**Live YOLO detection (`src/camera_detection.py`):** the "Revisión en Vivo (YOLO)" tab (`tab_camera_live` in app.py) uses `streamlit-webrtc` to stream the user's webcam (browser-side, so it works in Docker) and runs YOLOv8 (`ultralytics`, default `yolov8n.pt`, configurable via `YOLO_MODEL`) on each frame. `YOLOVideoProcessor.recv` draws boxes and, when a user-selected target class appears, calls `send_telegram_alert` (reusing the dashboard's synced `tg_id`) with a cooldown to avoid spam. The processor runs in a separate thread; config (target classes, confidence, chat_id) is injected from the main thread via `ctx.video_processor.*` on each rerun. `ultralytics`/`streamlit-webrtc` are imported lazily inside the tab so they don't slow app startup. The Dockerfile pre-downloads `yolov8n.pt` so detection works offline at runtime. Note: WebRTC works on localhost; remote access needs a TURN server.

**Telegram bot (`src/telegram_bot.py`):** `send_telegram_alert(message, chat_id)` is the one-shot push used by `app.py` when a prediction is unsafe. `run_listener()` is the long-running `python-telegram-bot` Application loop with `/start` and info commands; it owns its own asyncio loop inside the daemon thread.

## Gotchas

- `app.py` does `sys.path.append('src')` AND `from src.telegram_bot import ...` — the dual setup means `src/` modules can import each other as top-level (`import preprocessing`) when run directly, but app.py treats `src` as a package. Keep both import styles working.
- The Streamlit `@st.cache_resource` guard on the bot thread is load-bearing: removing it spawns a new bot on every rerun and Telegram will reject duplicate `getUpdates` calls.
- `requirements.txt` pins `numpy>=2.3` and `scikit-learn>=1.7` — pickled model/scaler in `models/` were trained against these versions; downgrading will likely break `joblib.load`.
