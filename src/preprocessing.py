"""
Módulo de Preprocesamiento y Escalado de Datos
Autor: Sebastian Yambay
Repositorio: https://github.com/SebastianYambay/Proyecto_Base_Conocimiento
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(file_path):
    """Carga el dataset desde una ruta especificada."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    df = pd.read_csv(file_path)
    print(f"-> Datos cargados desde {file_path} con dimensiones: {df.shape}")
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Divide el dataset en conjuntos de entrenamiento y prueba."""
    X = df.drop(columns=[target_column], axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_save_scaler(X_train, output_path='../models/scaler.pkl'):
    """Entrena el StandardScaler con los datos de entrenamiento y guarda el objeto.
    Retorna los datos escalados.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    joblib.dump(scaler, output_path)
    print(f"-> Escalador guardado en {output_path}")
    return X_train_scaled

def scale_data(X, scaler_path='../models/scaler.pkl'):
    """Carga un scaler existente y transforma nuevos datos."""
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    return X_scaled