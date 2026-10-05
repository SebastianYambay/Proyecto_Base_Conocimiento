"""
SIPCA — Sistema de Predicción de Calidad de Agua
Dashboard interactivo de Machine Learning, Visión por Computadora y Monitoreo.

Autor: Sebastian Yambay
GitHub: https://github.com/SebastianYambay/Proyecto_Base_Conocimiento
"""

import streamlit as st
import threading
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import json 
import sys
import os
import datetime
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Añadir src al path para poder importar
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.telegram_bot import send_telegram_alert, run_listener
from src.vision_module import analyze_water_turbidity, get_ntu_interpretation
from src.chatbot_llm import create_chatbot_widget

@st.cache_resource
def iniciar_bot_en_background():
    """
    Esta función crea un hilo secundario para correr el bot.
    Al usar @st.cache_resource, Streamlit asegura que esto solo se ejecute
    UNA vez al arrancar la app, evitando duplicar bots.
    """
    # Creamos el hilo apuntando a la función run_listener
    bot_thread = threading.Thread(target=run_listener, daemon=True)
    bot_thread.start()
    return bot_thread

# Llamamos a la función inmediatamente
iniciar_bot_en_background()

# Configuración inicial
st.set_page_config(
    page_title="SIPCA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');
    
    :root {
        --primary: var(--primary-color, #0c67a3);
        --accent: #11a4d4;
        --background: var(--background-color, #f0f4f8);
        --card: var(--secondary-background-color, #ffffff);
        --text-primary: var(--text-color, #101d22);
        --text-secondary: #5a6e79;
        --border-color: #e2e8f0;
    }
    
    /* Estilos generales */
    .stApp {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Iconos Material Symbols */
    .material-symbols-outlined {
        font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }
    
    /* Tarjeta de resultado personalizada */
    .result-card {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 1rem;
        border: 1px solid var(--border-color);
        padding: 3rem;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        margin: 2rem 0;
    }
    
    .result-icon {
        width: 96px;
        height: 96px;
        margin: 0 auto 1rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 60px;
    }
    
    .result-icon.potable {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
    }
    
    .result-icon.no-potable {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .result-title.potable {
        color: #22c55e;
    }
    
    .result-title.no-potable {
        color: #ef4444;
    }
    
    .result-confidence {
        color: var(--text-secondary);
        font-size: 1.125rem;
    }

    /* Sliders: Thumb (círculo) blanco con borde azul */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #ffffff !important;
        border: 2px solid var(--primary) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        height: 18px !important; 
        width: 18px !important;
    }
    
    /* Sliders: Track Fill (Barra rellena) - Forzar color */
    div[data-baseweb="slider"] > div > div > div > div {
        background-color: var(--primary) !important;
    }
    
    /* Botón Secundario (Reset): Fondo gris claro */
    button[kind="secondary"] {
        background-color: #f1f5f9 !important;
        border: 1px solid transparent !important;
        color: var(--text-secondary) !important;
        transition: all 0.2s;
    }
    button[kind="secondary"]:hover {
        background-color: #e2e8f0 !important;
        color: var(--text-primary) !important;
    }
    
    /* Botón Primario (Analizar): Azul con hover */
    button[kind="primary"] {
        background-color: var(--primary) !important;
        border: none !important;
        color: white !important;
        transition: all 0.2s;
    }
    button[kind="primary"]:hover {
        background-color: var(--accent) !important;
        box-shadow: 0 4px 12px rgba(12, 103, 163, 0.2);
    }

    /* Ocultar menú de Streamlit y footer */
    /*
    [data-testid="stToolbar"] {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    */

    /* --- NAVEGACIÓN SIDEBAR CON ICONOS --- */
    
    /* Ajuste del texto para alinear con el icono */
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem; /* Espacio entre icono y texto */
    }

    /* INYECCIÓN DE ICONOS MATERIAL SYMBOLS */
    
    /* 1. Dashboard General */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1) p::before {
        content: "dashboard";
        font-family: 'Material Symbols Outlined';
        font-size: 20px;
        font-weight: normal;
        font-variation-settings: 'FILL' 0;
    }
    /* Relleno cuando está activo */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1):has(input:checked) p::before {
        font-variation-settings: 'FILL' 1;
        color: var(--primary);
    }
    
    /* 2. Visión por Computadora */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2) p::before {
        content: "image_search";
        font-family: 'Material Symbols Outlined';
        font-size: 20px;
        font-weight: normal;
        font-variation-settings: 'FILL' 0;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2):has(input:checked) p::before {
        font-variation-settings: 'FILL' 1;
        color: var(--primary);
    }
    
    /* 3. Monitoreo de Cámaras */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(3) p::before {
        content: "videocam";
        font-family: 'Material Symbols Outlined';
        font-size: 20px;
        font-weight: normal;
        font-variation-settings: 'FILL' 0;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(3):has(input:checked) p::before {
        font-variation-settings: 'FILL' 1;
        color: var(--primary);
    }
    
    /* 4. Revisión en Vivo (YOLO) */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(4) p::before {
        content: "photo_camera";
        font-family: 'Material Symbols Outlined';
        font-size: 20px;
        font-weight: normal;
        font-variation-settings: 'FILL' 0;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(4):has(input:checked) p::before {
        font-variation-settings: 'FILL' 1;
        color: var(--primary);
    }
    
</style>
""", unsafe_allow_html=True)



# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/water_potability_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
# Cargar modelos y escalador

@st.cache_resource
def load_artifacts():
    """Carga el modelo y el escalador"""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception:
        st.error("Error: No se encontró el modelo o el escalador. Por favor, asegúrese de que los archivos existen en la ruta especificada.")
        return None, None

model, scaler = load_artifacts()

# Lista ordenada de variables por importancia
FEATURES_IMPORTANCE_ORDER = [
    'Sulfate', 'ph', 'Solids', 'Hardness', 'Chloramines',
    'Trihalomethanes', 'Turbidity', 'Conductivity', 'Organic_carbon',
]

# Sidebar con iconos Material Symbols
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
    <div style="display: flex; align-items: center; justify-content: center; width: 48px; height: 48px; background-color: #dbeafe; border-radius: 50%; color: #0369a1;">
        <span class="material-symbols-outlined" style="font-size: 28px; font-variation-settings: 'FILL' 1;">water_drop</span>
    </div>
    <div>
        <h1 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #0f172a; line-height: 1.2; font-family: 'Space Grotesk', sans-serif;">SIPCA</h1>
        <p style="margin: 0; font-size: 0.875rem; color: #64748b; font-weight: 500;">Predicción de potabilidad</p>
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown('---')

# Navegación
options = ["Dashboard General", "Análisis de Imágenes", "Monitoreo de Cámaras", "Revisión en Vivo (YOLO)"]
selection = st.sidebar.radio("Navegación", options, label_visibility="collapsed")
st.sidebar.markdown('---')

def tab_dashboard():
    # Definir los sliders con valores realistas o promedio
    def user_input_features():
        """Función para capturar los inputs del usuario a través de sliders"""
        st.sidebar.markdown('### Parámetros de la Muestra')
        
        # Agrupar parámetros para ahorrar espacio
        with st.sidebar.expander("Parámetros Básicos", expanded=True):
            ph = st.slider('pH', 0.0, 14.0, 7.0, 0.1)
            hardness = st.slider('Dureza (mg/L)', 50.0, 350.0, 196.0, 1.0)
            solids = st.slider('Sólidos (ppm)', 300.0, 60000.0, 22000.0, 100.0)
            chloramines = st.slider('Cloraminas (ppm)', 0.0, 14.0, 7.1, 0.1)

        with st.sidebar.expander("Parámetros Avanzados", expanded=False):
            sulfate = st.slider('Sulfato (mg/L)', 100.0, 500.0, 333.0, 1.0)
            conductivity = st.slider('Conductividad (µS/cm)', 100.0, 800.0, 420.0, 1.0)
            organic_carbon = st.slider('Carbono Orgánico (ppm)', 0.0, 30.0, 14.5, 0.1)
            trihalomethanes = st.slider('Trihalometanos', 0.0, 125.0, 66.0, 0.1)
            turbidity = st.slider('Turbidez', 1.0, 7.0, 3.9, 0.1)

        data = {
            'ph': ph,
            'Hardness': hardness,
            'Solids': solids,
            'Chloramines': chloramines,
            'Sulfate': sulfate,
            'Conductivity': conductivity,
            'Organic_carbon': organic_carbon,
            'Trihalomethanes': trihalomethanes,
            'Turbidity': turbidity
        }
        
        return pd.DataFrame([data], index=['Your Sample'])

    input_df = user_input_features()

    st.sidebar.markdown('---')
    with st.sidebar.expander("🔔 Conectar Alertas", expanded=True):
        # Enlace directo a tu bot
        bot_name = os.getenv("TELEGRAM_BOT_NAME", "analistaagua_bot") # Nombre real del bot (sin @)
        st.markdown(f"1. [Abrir Bot en Telegram](https://t.me/{bot_name}) y dar **/start**")
        
        if st.button("🔄 Sincronizar con Bot"):
            try:
                with open("telegram_connection.json", "r") as f:
                    data = json.load(f)
                
                # Guardar en sesión
                st.session_state['tg_id'] = data['chat_id']
                st.session_state['tg_name'] = data['name']
                st.success(f"Conectado: {data['name']}")
            except FileNotFoundError:
                st.warning("Primero ve a Telegram y usa /start")
                
        # Estado actual
        if 'tg_id' in st.session_state:
            st.caption(f"✅ Enviando a: {st.session_state['tg_name']}")
        else:
            st.caption("🔴 No conectado")
            
    # Botones de la barra lateral
    # st.sidebar.markdown('---')
    analyze_button = st.sidebar.button("Analizar Muestra", type="primary", width="stretch")
    st.sidebar.button("Restablecer Parámetros", type="secondary", width="stretch")

    # Área principal

    # --- Guía rápida / glosario del flujo ---
    with st.expander("📖 Guía rápida: ¿cómo funciona SIPCA?", expanded=True):
        st.markdown("""
**Flujo del Dashboard (predicción de potabilidad):**

1. **Ajusta los parámetros** de la muestra en la barra lateral ⬅️ (pH, dureza, sólidos, etc.).
2. Pulsa **«Analizar Muestra»** → el modelo de IA clasifica el agua como **Potable** o **No Potable** con un % de confianza.
3. Revisa los gráficos: **Importancia de Características** (qué pesa más en la decisión) y **Muestra vs Promedios Seguros** (tu muestra contra rangos ideales).
4. Si el agua es **No Potable** o el **pH está fuera de norma** (6.5–8.5), se envía una **alerta a Telegram** (si sincronizaste el bot).

**Para recibir alertas:** barra lateral → **🔔 Conectar Alertas** → abre el bot, dale **/start**, y pulsa **«Sincronizar con Bot»**.

**Análisis por lotes:** sube un **CSV** con varias muestras para predecirlas todas de una vez (abajo ⬇️).

---
**Las otras secciones del menú lateral:**
- 🖼️ **Análisis de Imágenes:** sube una foto de agua y la IA de visión estima la **turbidez (NTU)**.
- 📹 **Monitoreo de Cámaras:** panel de cámaras de vigilancia de fuentes de agua.
- 📷 **Revisión en Vivo (YOLO):** usa tu **webcam o el celular** para detectar objetos en tiempo real y disparar alertas.
- 🤖 **Asistente IA** (botón flotante abajo a la derecha): chatbot local para consultas sobre calidad de agua.
        """)

    # Bloque de análisis por lotes con icono Material Symbols
    with st.container(border=True):
        col_icon, col_text = st.columns([1, 15])
        with col_icon:
            st.markdown('<span class="material-symbols-outlined" style="font-size: 32px; color: var(--primary);">csv</span>', unsafe_allow_html=True)
        with col_text:
            st.markdown("### Análisis por lotes")
            st.caption("Sube un archivo CSV para realizar predicciones masivas. (Asegúrate de que las columnas coincidan con las esperadas a la muestra.)")
        
        csv_file = st.file_uploader(" ", type=["csv"], label_visibility="collapsed")

    if csv_file is not None:
        batch_df = pd.read_csv(csv_file)
        st.subheader("Preview de Archivo CSV")
        st.dataframe(batch_df.head())
        
        # Predicción de lotes
        if st.button("Ejecutar Predicción por Lotes", type="primary"):
            try:
                batch_scaled = scaler.transform(batch_df)
                predictions = model.predict(batch_scaled)
                batch_df['Potability_Prediction'] = np.where(predictions == 1, 'POTABLE', 'NO POTABLE')
                
                st.success("Análisis por lotes completado.")
                
                st.subheader("Preview de Resultados")
                st.dataframe(batch_df)

                st.download_button(
                    label="Descargar Resultados como CSV",
                    data=batch_df.to_csv(index=False).encode('utf-8'),
                    file_name='water_potability_results.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Error al procesar el lote: {e}. Asegurate de que las columnas coinciden con las esperadas.")

    # Predicción y resultados
    if analyze_button and model:
        # Preprocesamiento
        input_scaled = scaler.transform(input_df)
        
        # Predicción
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        confidence = proba[prediction] * 100
        
        prediction = model.predict(input_scaled)[0]
        ph_val = input_df['ph'].iloc[0]
        
        # === NUEVO: GUARDAR ESTADO PARA EL BOT (/status) ===
        status_data = {
            "prediction": "POTABLE" if prediction == 1 else "NO POTABLE",
            "ph": float(ph_val),
            "confidence": float(confidence),
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
        }
        with open("water_status.json", "w") as f:
            json.dump(status_data, f)
        
        # SISTEMA DE ALERTAS INTEGRADO
        trigger = False
        reasons = []
        
        # 1. Criterio IA
        if prediction == 0: # 0 = No Potable
            trigger = True
            reasons.append(f"IA detectó riesgo (Confianza: {confidence:.1f}%)")
            
        # 2. Criterio Normativo (pH)
        ph_val = input_df['ph'].iloc[0]
        if ph_val < 6.5 or ph_val > 8.5:
            trigger = True
            reasons.append(f"pH fuera de norma ({ph_val:.1f})")

        # 3. Disparo de Alerta
        if trigger:
            # Recuperar ID de la sesión
            chat_id = st.session_state.get('tg_id')
            
            if chat_id:
                msg = (
                    f"🚨 *ALERTA DE CALIDAD DE AGUA*\n\n"
                    f"**Motivos:** {', '.join(reasons)}\n"
                    f"**Muestra:** pH {ph_val:.1f}"
                )
                # Llamamos a la función que importamos de src/telegram_bot.py
                ok, status = send_telegram_alert(msg, chat_id)
                
                if ok:
                    st.toast(f"Alerta enviada a {st.session_state['tg_name']}", icon="📲")
                else:
                    st.error(f"Fallo Telegram: {status}")
            else:
                st.warning("⚠️ Riesgo detectado, pero no has sincronizado el Bot.")
        
        # Mostrar resultados con diseño del mockup
        if prediction == 1:
            icon_class = "potable"
            icon_symbol = "check_circle"
            title_text = "Potable"
            title_class = "potable"
        else:
            icon_class = "no-potable"
            icon_symbol = "cancel"
            title_text = "NO Potable"
            title_class = "no-potable"
        
        st.markdown(f"""
        <div class="result-card">
            <div class="result-icon {icon_class}">
                <span class="material-symbols-outlined">{icon_symbol}</span>
            </div>
            <h3 class="result-title {title_class}">{title_text}</h3>
            <p class="result-confidence">{confidence:.1f}% Confianza</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualizaciones
        col_feat_imp, col_radar = st.columns([3, 2])
        
        # Gráfico1: Importancia de características (REAL, desde el modelo)
        with col_feat_imp:
            st.subheader("Importancia de Características")

            # Orden EXACTO de las features con que se entrenó el modelo
            # (mismo orden que input_df, que es lo que recibe el scaler/modelo)
            feature_names = [
                'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
            ]

            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
                df_imp = pd.DataFrame({
                    'Característica': feature_names,
                    'Importancia': importance_values
                }).sort_values(by='Importancia', ascending=True)

                # Gráfico de barras horizontales con color accent del diseño
                st.bar_chart(df_imp, x='Importancia', y='Característica', color='#11a4d4', height=400)
                st.caption("Importancia real aprendida por el modelo RandomForest.")
            else:
                st.info("El modelo cargado no expone importancias de características.")

        # Gráfico Radar Chart
        with col_radar:
            st.subheader("Muestra vs Promedios Seguros")
            
            # Valores promedios seguros estimados para la comparación
            safe_avg_values = {
                    'ph': 7.5, 'Hardness': 180.0, 'Solids': 15000.0, 'Chloramines': 5.0, 
                    'Sulfate': 300.0, 'Conductivity': 500.0, 'Organic_carbon': 10.0, 
                    'Trihalomethanes': 70.0, 'Turbidity': 4.0
            }
            
            # Asegurar el orden del sample coincida con el safe_avg
            sample_values = input_df[list(safe_avg_values.keys())].iloc[0].values.tolist()
            safe_values = list(safe_avg_values.values())
            categories = list(safe_avg_values.keys())
            
            # Encontrar el valor máximo para establecer el rango del eje polar
            max_val = max(max(sample_values), max(safe_values))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=sample_values,
                theta=categories,
                fill='toself',
                name='Tu Muestra',
                line_color='#11a4d4',
                fillcolor='rgba(17, 164, 212, 0.4)'
            ))
            
            # Promedios seguros 
            fig.add_trace(go.Scatterpolar(
                r=safe_values,
                theta=categories,
                fill='none',
                name='Promedios Seguros',
                line=dict(dash='dot', color='#4ade80')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max_val * 1.1]),
                    angularaxis=dict(tickfont=dict(size=10), direction = "clockwise")
                ),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, width="stretch")

def tab_vision():
    st.header("Análisis de Imágenes")
    st.caption("Análisis de turbidez mediante visión por computadora. Sube una imagen de tu muestra de agua.")
    
    # Información educativa en un expander
    with st.expander("ℹ️ ¿Qué es la Turbidez (NTU)?", expanded=False):
        st.markdown("""
        La **turbidez** mide la claridad del agua. Se expresa en **NTU** (Unidades Nefelométricas de Turbidez).
        
        Partículas suspendidas como arcilla, sedimentos, algas y microorganismos causan turbidez.
        
        **Estándares:**
        - 🌟 **0-1 NTU**: Excelente (agua cristalina)
        - ✅ **1-5 NTU**: Cumple con OMS (potable)
        - ⚠️ **5-10 NTU**: Aceptable (visible al ojo)
        - 🔶 **10-25 NTU**: Deficiente (requiere tratamiento)
        - 🚫 **>25 NTU**: No potable
        """)
    
    # Layout de dos columnas
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        # Zona de carga con diseño mejorado
        with st.container(border=True):
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <span class="material-symbols-outlined" style="font-size: 48px; color: var(--primary);">add_photo_alternate</span>
                <h3 style="margin: 0.5rem 0;">Sube una imagen de agua</h3>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">Formatos: JPG, PNG, JPEG</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Seleccionar imagen",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
    
    with col_info:
        # Guía rápida
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%); 
                    padding: 1.5rem; border-radius: 0.75rem; height: 100%;">
            <h4 style="margin-top: 0;">📸 Consejos de Fotografía</h4>
            <ul style="font-size: 0.9rem; line-height: 1.8;">
                <li>Usa luz natural uniforme</li>
                <li>Fondo blanco o neutro</li>
                <li>Recipiente transparente</li>
                <li>Enfoque nítido</li>
                <li>Sin reflejos directos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Leer los bytes de la imagen
        image_bytes = uploaded_file.read()
        
        # Mostrar imagen subida
        col_img, col_result = st.columns([1, 1])
        
        with col_img:
            st.subheader("Muestra Analizada")
            st.image(image_bytes, width="stretch", caption=uploaded_file.name)
        
        with col_result:
            st.subheader("Resultados del Análisis")
            
            # Botón de análisis
            if st.button("🔬 Analizar Turbidez", type="primary", width="stretch"):
                with st.spinner("Analizando imagen..."):
                    # Llamar a la función de visión
                    result = analyze_water_turbidity(image_bytes)
                    
                    if result.get('error'):
                        st.error(result['message'])
                    else:
                        # Guardar resultado en session_state
                        st.session_state['vision_result'] = result
                        st.rerun()
        
        # Mostrar resultados si existen
        if 'vision_result' in st.session_state:
            result = st.session_state['vision_result']
            
            # Determinar estilo según el estado
            if result['status'] == 'safe':
                icon_class = "potable"
                icon_symbol = "water_drop"
                border_color = "#22c55e"
            elif result['status'] == 'acceptable':
                icon_class = "no-potable"
                icon_symbol = "opacity"
                border_color = "#f59e0b"
            else:
                icon_class = "no-potable"
                icon_symbol = "warning"
                border_color = "#ef4444"
            
            st.markdown("---")
            
            # Tarjeta de resultado principal
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.6);
                        border-radius: 1rem;
                        border: 3px solid {border_color};
                        padding: 2rem;
                        text-align: center;
                        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
                        margin: 2rem 0;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 2rem;">
                    <div class="result-icon {icon_class}">
                        <span class="material-symbols-outlined">{icon_symbol}</span>
                    </div>
                    <div style="text-align: left;">
                        <h1 style="margin: 0; font-size: 3rem; font-weight: bold; color: {border_color};">{result['ntu']} NTU</h1>
                        <h3 style="margin: 0.5rem 0; color: var(--text-primary);">{result['classification']}</h3>
                        <p style="margin: 0; color: var(--text-secondary);">Confianza: {result['confidence']}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recomendación
            st.info(result['recommendation'])
            
            # Indicador de cumplimiento OMS
            if result['meets_who_standards']:
                st.success("✅ Cumple con los estándares de la OMS (< 5 NTU)")
            else:
                st.warning("⚠️ No cumple con los estándares recomendados de la OMS")
            
            # Mostrar insignia de "Powered by AI"
            if result.get('powered_by'):
                st.caption(f"🤖 Análisis realizado con {result['powered_by']}")
            
            # Observaciones visuales del AI
            st.markdown("### 👁️ Observaciones Visuales (AI)")
            color_profile = result.get('color_profile', {})
            
            obs_cols = st.columns(4)
            with obs_cols[0]:
                st.metric(
                    label="Claridad",
                    value=color_profile.get('clarity', 'N/A')[:15],
                    help="Evaluación de transparencia del agua"
                )
            
            with obs_cols[1]:
                st.metric(
                    label="Tinte de Color",
                    value=color_profile.get('color_tint', 'N/A')[:15],
                    help="Coloración observada en la muestra"
                )
            
            with obs_cols[2]:
                st.metric(
                    label="Partículas Visibles",
                    value=color_profile.get('visible_particles', 'N/A')[:15],
                    help="Cantidad de partículas suspendidas"
                )
            
            with obs_cols[3]:
                st.metric(
                    label="Transmisión de Luz",
                    value=color_profile.get('light_transmission', 'N/A')[:15],
                    help="Capacidad del agua para transmitir luz"
                )
            
            # Insights adicionales del AI
            if 'ai_insights' in result:
                ai_insights = result['ai_insights']
                
                # Indicadores de calidad
                st.markdown("### 🔬 Indicadores de Calidad")
                quality_indicators = ai_insights.get('quality_indicators', {})
                
                ind_cols = st.columns(3)
                with ind_cols[0]:
                    st.info(f"**Sólidos Suspendidos:** {quality_indicators.get('suspended_solids', 'N/A')}")
                with ind_cols[1]:
                    st.info(f"**Presencia de Sedimento:** {quality_indicators.get('sediment_presence', 'N/A')}")
                with ind_cols[2]:
                    st.info(f"**Materia Orgánica:** {quality_indicators.get('organic_matter', 'N/A')}")
                
                # Posibles causas
                potential_causes = ai_insights.get('potential_causes', [])
                if potential_causes:
                    st.markdown("### 🔍 Posibles Causas de Turbidez")
                    for cause in potential_causes:
                        st.markdown(f"- {cause}")
                
                # Notas sobre calidad de imagen
                image_notes = ai_insights.get('image_quality_notes', '')
                if image_notes:
                    with st.expander("📸 Notas sobre Calidad de Imagen"):
                        st.caption(image_notes)
            
            # Gráfico de referencia NTU
            st.markdown("### 📈 Escala de Referencia NTU")
            
            # Crear gráfico de barras horizontal con la escala
            ntu_ranges = [
                {'label': 'Excelente\n(0-1)', 'value': 1, 'color': '#22c55e'},
                {'label': 'Muy Buena\n(1-5)', 'value': 4, 'color': '#84cc16'},
                {'label': 'Buena\n(5-10)', 'value': 5, 'color': '#eab308'},
                {'label': 'Aceptable\n(10-25)', 'value': 15, 'color': '#f59e0b'},
                {'label': 'Deficiente\n(25-50)', 'value': 25, 'color': '#ef4444'},
                {'label': 'Muy Turbia\n(>50)', 'value': 50, 'color': '#991b1b'}
            ]
            
            fig = go.Figure()
            
            # Agregar barras de referencia
            for i, range_data in enumerate(ntu_ranges):
                fig.add_trace(go.Bar(
                    y=[range_data['label']],
                    x=[range_data['value']],
                    orientation='h',
                    marker=dict(color=range_data['color']),
                    name=range_data['label'],
                    showlegend=False
                ))
            
            # Agregar marcador de tu muestra
            y_position = 0
            for i, range_data in enumerate(ntu_ranges):
                cumulative = sum(r['value'] for r in ntu_ranges[:i+1])
                if result['ntu'] <= cumulative:
                    y_position = i
                    break
            
            fig.add_trace(go.Scatter(
                x=[result['ntu']],
                y=[ntu_ranges[y_position]['label']],
                mode='markers',
                marker=dict(
                    size=20,
                    color='white',
                    symbol='diamond',
                    line=dict(color='black', width=2)
                ),
                name='Tu Muestra',
                showlegend=True
            ))
            
            fig.update_layout(
                barmode='overlay',
                xaxis_title='NTU',
                height=400,
                showlegend=True,
                legend=dict(x=0.8, y=0.95),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Botón para nuevo análisis
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 Analizar Nueva Imagen", width="stretch"):
                    del st.session_state['vision_result']
                    st.rerun()
            
            with col_btn2:
                # Exportar resultados
                result_text = f"""
=== ANÁLISIS DE TURBIDEZ ===
Imagen: {uploaded_file.name}
Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULTADO:
Turbidez: {result['ntu']} NTU
Clasificación: {result['classification']}
Confianza: {result['confidence']}%

RECOMENDACIÓN:
{result['recommendation']}

Cumple OMS: {'Sí' if result['meets_who_standards'] else 'No'}
                """
                st.download_button(
                    label="📄 Exportar Reporte",
                    data=result_text,
                    file_name=f"turbidity_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    width="stretch"
                )
    else:
        # Estado inicial sin imagen
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: var(--text-secondary);">
            <span class="material-symbols-outlined" style="font-size: 80px; opacity: 0.3;">image</span>
            <h3 style="opacity: 0.6;">Esperando imagen...</h3>
            <p>Sube una imagen para comenzar el análisis de turbidez</p>
        </div>
        """, unsafe_allow_html=True)

def tab_cameras():
    """Tab de monitoreo de cámaras en tiempo real"""
    st.header("Monitoreo de Cámaras en Tiempo Real")
    st.caption("Sistema de vigilancia inteligente para fuentes de agua en comunidades rurales")
    
    # Cargar datos de cámaras
    cameras_file = os.path.join(BASE_DIR, "cameras/info.json")
    try:
        with open(cameras_file, 'r', encoding='utf-8') as f:
            cameras_data = json.load(f)
    except Exception as e:
        st.error(f"Error al cargar datos de cámaras: {str(e)}")
        return
    
    # Panel de control superior con estadísticas
    st.markdown("### 📊 Panel de Control")
    
    # Calcular estadísticas
    total_cameras = len(cameras_data)
    online_cameras = sum(1 for cam in cameras_data if cam['status'] == 'online')
    total_detections_today = sum(cam['daily_detections'] for cam in cameras_data)
    high_alerts = sum(1 for cam in cameras_data if cam['alert_level'] in ['high', 'critical'])
    
    # Métricas principales
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric(
            label="🎥 Cámaras Activas",
            value=f"{online_cameras}/{total_cameras}",
            delta="100%" if online_cameras == total_cameras else f"{int(online_cameras/total_cameras*100)}%"
        )
    
    with metrics_cols[1]:
        st.metric(
            label="🔍 Detecciones Hoy",
            value=total_detections_today,
            delta="+12 vs ayer"
        )
    
    with metrics_cols[2]:
        st.metric(
            label="⚠️ Alertas Altas",
            value=high_alerts,
            delta=f"{high_alerts} activas",
            delta_color="inverse"
        )
    
    with metrics_cols[3]:
        avg_quality = sum(cam['water_quality']['turbidity_ntu'] for cam in cameras_data) / len(cameras_data)
        st.metric(
            label="💧 Turbidez Promedio",
            value=f"{avg_quality:.1f} NTU",
            delta="Monitoreo continuo"
        )
    
    st.markdown("---")
    
    # Filtros
    col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 2])
    
    with col_filter1:
        filter_alert = st.selectbox(
            "🚨 Nivel de Alerta",
            ["Todos", "Crítico", "Alto", "Medio", "Bajo"],
            index=0
        )
    
    with col_filter2:
        filter_location = st.selectbox(
            "📍 Ubicación",
            ["Todas"] + list(set(cam['location'] for cam in cameras_data)),
            index=0
        )
    
    with col_filter3:
        sort_by = st.selectbox(
            "🔽 Ordenar por",
            ["Última actualización", "Nivel de alerta", "Detecciones", "Turbidez"],
            index=0
        )
    
    # Aplicar filtros
    filtered_cameras = cameras_data.copy()
    
    if filter_alert != "Todos":
        alert_map = {"Crítico": "critical", "Alto": "high", "Medio": "medium", "Bajo": "low"}
        filtered_cameras = [cam for cam in filtered_cameras if cam['alert_level'] == alert_map[filter_alert]]
    
    if filter_location != "Todas":
        filtered_cameras = [cam for cam in filtered_cameras if cam['location'] == filter_location]
    
    # Ordenar
    if sort_by == "Nivel de alerta":
        alert_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        filtered_cameras.sort(key=lambda x: alert_order.get(x['alert_level'], 4))
    elif sort_by == "Detecciones":
        filtered_cameras.sort(key=lambda x: x['daily_detections'], reverse=True)
    elif sort_by == "Turbidez":
        filtered_cameras.sort(key=lambda x: x['water_quality']['turbidity_ntu'], reverse=True)
    
    st.markdown("---")
    
    # Mostrar cámaras
    if not filtered_cameras:
        st.info("🔍 No se encontraron cámaras con los filtros seleccionados")
        return
    
    for camera in filtered_cameras:
        # Determinar color de alerta
        alert_colors = {
            'low': {'bg': '#d1fae5', 'border': '#22c55e', 'icon': 'check_circle', 'text': 'Bajo'},
            'medium': {'bg': '#fef3c7', 'border': '#f59e0b', 'icon': 'warning', 'text': 'Medio'},
            'high': {'bg': '#fee2e2', 'border': '#ef4444', 'icon': 'error', 'text': 'Alto'},
            'critical': {'bg': '#fecaca', 'border': '#991b1b', 'icon': 'dangerous', 'text': 'Crítico'}
        }
        
        alert_info = alert_colors.get(camera['alert_level'], alert_colors['low'])
        
        with st.container(border=True):
            # Header de la cámara
            col_header_left, col_header_right = st.columns([3, 1])
            
            with col_header_left:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span class="material-symbols-outlined" style="font-size: 32px; color: var(--primary);">videocam</span>
                    <div>
                        <h3 style="margin: 0; font-size: 1.25rem;">{camera['name']}</h3>
                        <p style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">
                            <span class="material-symbols-outlined" style="font-size: 16px; vertical-align: middle;">location_on</span>
                            {camera['location']} • ID: {camera['camera_id']}
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_header_right:
                st.markdown(f"""
                <div style="text-align: right;">
                    <div style="display: inline-flex; align-items: center; gap: 0.5rem; 
                                background: {alert_info['bg']}; border: 2px solid {alert_info['border']};
                                padding: 0.5rem 1rem; border-radius: 2rem;">
                        <span class="material-symbols-outlined" style="font-size: 20px; color: {alert_info['border']};">{alert_info['icon']}</span>
                        <span style="font-weight: bold; color: {alert_info['border']};">Alerta {alert_info['text']}</span>
                    </div>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: var(--text-secondary);">
                        <span class="material-symbols-outlined" style="font-size: 14px;">schedule</span>
                        {camera['last_update']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Cuerpo principal - Imagen y datos
            col_image, col_data = st.columns([2, 3])
            
            with col_image:
                # Mostrar imagen de cámara
                img_path = os.path.join(BASE_DIR, camera['img'])
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True, caption="Feed en Tiempo Real")
                else:
                    st.info("📷 Imagen no disponible")
                
                # Estado de conexión
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem;">
                    <div style="width: 10px; height: 10px; border-radius: 50%; background: #22c55e;"></div>
                    <span style="font-size: 0.9rem; color: var(--text-secondary);">
                        En línea • {camera['avg_response_time']} tiempo respuesta
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            with col_data:
                # Descripción
                st.markdown(f"**Descripción:** {camera['description']}")
                
                # Calidad del agua
                st.markdown("#### 💧 Parámetros de Calidad")
                quality = camera['water_quality']
                
                quality_cols = st.columns(3)
                with quality_cols[0]:
                    turbidity_color = "#22c55e" if quality['turbidity_ntu'] < 5 else "#f59e0b" if quality['turbidity_ntu'] < 10 else "#ef4444"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0.75rem; background: {turbidity_color}20; border-radius: 0.5rem;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: {turbidity_color};">{quality['turbidity_ntu']}</div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">NTU</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with quality_cols[1]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0.75rem; background: #e0f2fe; border-radius: 0.5rem;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: #0369a1;">{quality['temperature_c']}</div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">°C</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with quality_cols[2]:
                    ph_color = "#22c55e" if 6.5 <= quality['ph'] <= 8.5 else "#f59e0b"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0.75rem; background: {ph_color}20; border-radius: 0.5rem;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: {ph_color};">{quality['ph']}</div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">pH</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Objetos detectados
                st.markdown("#### 🔍 Objetos Detectados Hoy")
                
                if not camera['objects_detected']:
                    st.success("✅ No se detectaron contaminantes")
                else:
                    for obj in camera['objects_detected']:
                        # Determinar color según riesgo
                        risk_colors = {
                            'low': '#22c55e',
                            'medium': '#f59e0b',
                            'high': '#ef4444',
                            'critical': '#991b1b'
                        }
                        risk_color = risk_colors.get(obj['risk_level'], '#6b7280')
                        
                        # Calcular ancho de barra de confianza
                        confidence_width = int(obj['confidence'] * 100)
                        
                        st.markdown(f"""
                        <div style="margin: 0.5rem 0; padding: 0.75rem; background: {risk_color}10; 
                                    border-left: 4px solid {risk_color}; border-radius: 0.25rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                                <span style="font-weight: 600;">{obj['display_name']}</span>
                                <span style="background: {risk_color}; color: white; padding: 0.2rem 0.5rem; 
                                             border-radius: 0.25rem; font-size: 0.75rem;">
                                    {obj['risk_level'].upper()}
                                </span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem;">
                                <span>Confianza: {confidence_width}%</span>
                                <div style="flex: 1; height: 6px; background: #e5e7eb; border-radius: 3px; overflow: hidden;">
                                    <div style="width: {confidence_width}%; height: 100%; background: {risk_color};"></div>
                                </div>
                                <span style="color: var(--text-secondary);">Cantidad: {obj['count']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Footer con acciones
            st.markdown("<br>", unsafe_allow_html=True)
            action_cols = st.columns([1, 1, 1, 1])
            
            with action_cols[0]:
                if st.button("📊 Ver Historial", key=f"hist_{camera['camera_id']}", use_container_width=True):
                    st.info("Funcionalidad de historial próximamente")
            
            with action_cols[1]:
                if st.button("📸 Capturar Imagen", key=f"cap_{camera['camera_id']}", use_container_width=True):
                    st.success("Imagen capturada")
            
            with action_cols[2]:
                if st.button("🔔 Configurar Alertas", key=f"alert_{camera['camera_id']}", use_container_width=True):
                    st.info("Panel de configuración próximamente")
            
            with action_cols[3]:
                if st.button("📍 Ver en Mapa", key=f"map_{camera['camera_id']}", use_container_width=True):
                    coords = camera.get('coordinates', {})
                    if coords:
                        st.map(pd.DataFrame([coords]), zoom=13)

def tab_camera_live():
    """Revisión de cámaras en vivo: webcam + detección YOLO + alerta Telegram."""
    st.header("Revisión de Cámaras en Vivo (YOLO)")
    st.caption("Detección de objetos en tiempo real desde tu webcam. Si YOLO reconoce una clase objetivo, se envía una alerta de Telegram.")

    # Importación perezosa: evita cargar torch/ultralytics al arrancar la app
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
        from src.camera_detection import YOLOVideoProcessor, get_class_names
    except Exception as e:
        st.error(
            "Faltan dependencias para esta función.\n\n"
            "Instala con: `pip install ultralytics streamlit-webrtc av`\n\n"
            f"Detalle: {e}"
        )
        return

    # Cargar nombres de clases del modelo (descarga el modelo la 1ª vez)
    try:
        with st.spinner("Cargando modelo YOLO..."):
            class_names = get_class_names()
    except Exception as e:
        st.error(f"No se pudo cargar el modelo YOLO: {e}")
        return

    # --- Configuración en la barra lateral ---
    with st.sidebar.expander("⚙️ Configuración de Detección", expanded=True):
        default_targets = [c for c in ["bottle", "cup"] if c in class_names]
        target_classes = st.multiselect(
            "Objetos que disparan alerta",
            options=class_names,
            default=default_targets,
            help="Clases COCO. Por defecto: botellas y vasos (contaminantes plásticos)."
        )
        conf = st.slider("Confianza mínima", 0.1, 0.9, 0.5, 0.05)
        cooldown = st.slider("Enfriamiento entre alertas (seg)", 5, 120, 30, 5)

    # Auto-sincronizar con Telegram desde el archivo de conexión (si existe),
    # para que las alertas se envíen sin tener que pulsar "Sincronizar" manualmente.
    if 'tg_id' not in st.session_state:
        try:
            with open("telegram_connection.json", "r") as f:
                _conn = json.load(f)
            st.session_state['tg_id'] = _conn['chat_id']
            st.session_state['tg_name'] = _conn.get('name', 'Usuario')
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass

    # Estado de la conexión con Telegram
    chat_id = st.session_state.get('tg_id')
    if chat_id:
        st.success(f"📲 Las alertas se enviarán a: {st.session_state.get('tg_name', '?')}")
    else:
        st.warning("⚠️ No hay bot sincronizado. Abre el bot en Telegram y envía /start; luego recarga esta página.")

    if not target_classes:
        st.caption("ℹ️ No has seleccionado clases objetivo: se detectará todo, pero no se dispararán alertas.")

    # --- Selección de fuente de video ---
    source = st.radio(
        "Fuente de video",
        ["💻 Webcam del navegador", "📱 Cámara IP (celular)"],
        horizontal=True,
    )

    if source == "💻 Webcam del navegador":
        # Servidor STUN público para la negociación WebRTC
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        ctx = webrtc_streamer(
            key="yolo-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=YOLOVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Inyectar la configuración actual al procesador (en cada rerun)
        if ctx.video_processor:
            ctx.video_processor.target_classes = set(target_classes)
            ctx.video_processor.conf_threshold = conf
            ctx.video_processor.alert_cooldown = cooldown
            ctx.video_processor.chat_id = chat_id

        st.info("▶️ Pulsa **START** para activar tu webcam. Las cajas sobre el video son detecciones de YOLO en tiempo real.")
    else:
        _run_ip_camera(set(target_classes), conf, cooldown, chat_id)


def _run_ip_camera(target_classes, conf, cooldown, chat_id):
    """
    Lee el stream de una cámara IP (app tipo 'IP Webcam' en el celular) con
    OpenCV, corre YOLO y muestra el video procesado.

    Patrón Streamlit: un único bucle continuo actualiza SIEMPRE el mismo
    placeholder (st.empty) en sitio, sin st.rerun() -> video fluido, sin
    parpadeo. Al desmarcar el checkbox, Streamlit interrumpe el bucle (en la
    siguiente llamada a la API), y el bloque finally libera la cámara.
    """
    import time
    import cv2
    from src.camera_detection import detect_on_frame
    from src.telegram_bot import send_telegram_alert

    st.markdown(
        "**Cómo obtener la URL:** instala *IP Webcam* (Android) o *Larix Broadcaster*, "
        "inícialo y usa la URL que muestra, p. ej. `http://192.168.1.50:8080/video` (MJPEG). "
        "El celular y este equipo deben estar en la misma red."
    )

    default_url = st.session_state.get("ipcam_url", "http://192.168.1.50:8080/video")
    url = st.text_input("URL del stream de la cámara IP", value=default_url, key="ipcam_url")
    run = st.checkbox("▶️ Iniciar cámara IP", value=False, key="ipcam_run")

    info_box = st.empty()
    frame_box = st.empty()

    if not run:
        info_box.info("Marca **Iniciar cámara IP** para conectar con el celular.")
        return

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        info_box.error(f"No se pudo abrir el stream:\n`{url}`\n\nVerifica la URL y que el celular esté transmitiendo en la misma red.")
        return

    last_alert = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                info_box.warning("No se reciben frames. Revisa la URL y que la app del celular siga transmitiendo.")
                time.sleep(0.3)
                continue

            # Detección YOLO (st.image acepta BGR directamente con channels="BGR")
            annotated, detections = detect_on_frame(frame, conf)
            frame_box.image(annotated, channels="BGR", use_container_width=True)

            if detections:
                info_box.caption("🔍 " + ", ".join(f"{lbl} ({c:.0%})" for lbl, c in detections))
            else:
                info_box.caption("Sin detecciones")

            # Alerta si aparece una clase objetivo (con enfriamiento)
            hits = sorted({lbl for lbl, _ in detections if lbl in target_classes})
            if hits and chat_id:
                now = time.time()
                if now - last_alert >= cooldown:
                    last_alert = now
                    msg = (
                        "🚨 *ALERTA CÁMARA IP (YOLO)*\n\n"
                        f"Objeto(s) detectado(s): {', '.join(hits)}\n"
                        f"Hora: {time.strftime('%H:%M:%S')}"
                    )
                    send_telegram_alert(msg, chat_id)
                    st.toast(f"Alerta enviada: {', '.join(hits)}", icon="📲")
    finally:
        cap.release()


def tab_chatbot():
    st.header("Asistente de IA")
    st.caption("Haz preguntas sobre métricas de calidad del agua o recibe ayuda con el análisis.")
    st.info("Módulo en desarrollo. Aquí se implementará el chatbot.")


if selection == "Dashboard General":
    tab_dashboard()
elif selection == "Análisis de Imágenes":
    tab_vision()
elif selection == "Monitoreo de Cámaras":
    tab_cameras()
elif selection == "Revisión en Vivo (YOLO)":
    tab_camera_live()
create_chatbot_widget()

# Pie de página en barra lateral
st.sidebar.markdown('---')
st.sidebar.markdown(
    """
    <div style='text-align: center; color: #64748b; font-size: 0.78rem;'>
        <b>SIPCA</b> | Calidad de Agua<br>
        Desarrollado por <b>Sebastian Yambay</b><br>
        <a href='https://github.com/SebastianYambay/Proyecto_Base_Conocimiento' target='_blank' style='color: #0369a1; text-decoration: none;'>GitHub: Proyecto_Base_Conocimiento</a>
    </div>
    """,
    unsafe_allow_html=True
)