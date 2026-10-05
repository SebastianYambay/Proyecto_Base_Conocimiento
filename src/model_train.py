"""
Entrenamiento del Modelo Random Forest para Potabilidad del Agua
Autor: Sebastian Yambay
Repositorio: https://github.com/SebastianYambay/Proyecto_Base_Conocimiento
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os
import preprocessing as prep

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/processed/water_potability_cleaned.csv')
MODEL_PATH = os.path.join(BASE_DIR, '../models/water_potability_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '../models/scaler.pkl')

def train():
    print("Iniciando entrenamiento del modelo...")
    
    # 1. Cargar datos
    try:
        df = prep.load_data(DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo de datos en {DATA_PATH}.")
        print(e)
        return
    
    # 2. Dividir datos
    print("Separando datos en Train/Test...")
    X_train, X_test, y_train, y_test = prep.split_data(df, target_column='Potability')
    
    # 3. Escalar datos
    print("Escalando datos...")
    X_train_scaled = prep.train_save_scaler(X_train, output_path=SCALER_PATH)
    X_test_scaled = prep.scale_data(X_test, scaler_path=SCALER_PATH)
    
    # 4. Definir modelo
    print("Entrenando el modelo RandomForestClassifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    
    # 5. Entrenar modelo
    rf_model.fit(X_train_scaled, y_train)
    
    # 6. Evaluar modelo
    y_pred = rf_model.predict(X_test_scaled)
    y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print("Resultados del modelo:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # 7. Guardar modelo
    joblib.dump(rf_model, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    train()