# 📹 Sistema de Monitoreo de Cámaras

## Descripción
Sistema de vigilancia inteligente en tiempo real para fuentes de agua en comunidades rurales. Utiliza detección de objetos para identificar contaminantes y generar alertas automáticas.

## Características

### 🎥 Monitoreo en Tiempo Real
- Feed visual de 5 cámaras estratégicamente ubicadas
- Estado de conexión en vivo
- Última actualización timestamp

### 🔍 Detección Inteligente de Objetos
El sistema detecta y clasifica automáticamente:
- **Contaminantes sólidos**: Botellas plásticas, latas, bolsas, residuos
- **Contaminantes líquidos**: Aceite flotante, espuma química
- **Contaminantes orgánicos**: Algas, vegetación, sedimentos
- **Confianza de detección**: 0-100%
- **Nivel de riesgo**: Bajo, Medio, Alto, Crítico

### 💧 Parámetros de Calidad
Cada cámara monitorea:
- **Turbidez (NTU)**: Indicador de claridad del agua
- **Temperatura (°C)**: Temperatura del agua
- **pH**: Nivel de acidez/alcalinidad

### ⚠️ Sistema de Alertas
Clasificación por niveles:
- 🟢 **Bajo**: Sin contaminantes o mínima presencia
- 🟡 **Medio**: Contaminantes detectados, requiere atención
- 🟠 **Alto**: Múltiples contaminantes, acción inmediata
- 🔴 **Crítico**: Situación grave, respuesta urgente

### 📊 Panel de Control
- Total de cámaras activas
- Detecciones diarias agregadas
- Alertas activas
- Turbidez promedio de todas las ubicaciones

### 🔧 Funcionalidades
- **Filtros avanzados**: Por nivel de alerta, ubicación
- **Ordenamiento**: Por fecha, prioridad, detecciones, turbidez
- **Historial**: Registro de detecciones y eventos
- **Capturas**: Snapshots bajo demanda
- **Geolocalización**: Visualización en mapa

## Estructura de Datos

### info.json Schema
```json
{
  "camera_id": "CAM-XXX",
  "name": "Nombre descriptivo",
  "img": "cameras/CameraX.png",
  "location": "Ubicación física",
  "coordinates": {"lat": -0.xxxx, "lng": -78.xxxx},
  "description": "Descripción del punto de monitoreo",
  "status": "online|offline",
  "last_update": "ISO 8601 timestamp",
  "water_quality": {
    "turbidity_ntu": float,
    "temperature_c": float,
    "ph": float
  },
  "alert_level": "low|medium|high|critical",
  "objects_detected": [
    {
      "object_type": "tipo_objeto",
      "display_name": "Nombre legible",
      "confidence": 0.0-1.0,
      "risk_level": "low|medium|high|critical",
      "count": int,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "daily_detections": int,
  "avg_response_time": "string"
}
```

## Ubicaciones Actuales

1. **CAM-001**: Estanque Comunitario A - San Pedro
2. **CAM-002**: Río Alimentador B - Planta Norte
3. **CAM-003**: Lago Urbano Central - Sector C
4. **CAM-004**: Canal Desagüe Principal - Planta Sur
5. **CAM-005**: Reservorio Comunitario E - La Esperanza

## Integración Futura

### Modelo de Detección de Objetos
Para implementar detección real:
```python
# Usar YOLO, Faster R-CNN o similares
from ultralytics import YOLO

model = YOLO('water_contaminants_detector.pt')
results = model(image)

# Procesar detecciones
for detection in results:
    object_type = detection.class_name
    confidence = detection.confidence
    bbox = detection.bbox
```

### API de Streaming
```python
# WebSocket para feeds en vivo
@app.websocket("/camera/{camera_id}")
async def camera_stream(websocket, camera_id):
    while True:
        frame = capture_frame(camera_id)
        detections = detect_objects(frame)
        await websocket.send_json({
            "frame": encode_frame(frame),
            "detections": detections
        })
```

## Mantenimiento

### Agregar Nueva Cámara
1. Añadir entrada en `info.json`
2. Colocar imagen en `cameras/CameraX.png`
3. Definir coordenadas GPS
4. Configurar umbrales de alerta

### Actualizar Detecciones
Modificar `objects_detected` array con nuevos objetos detectados por el sistema de visión artificial.

## Tecnologías Sugeridas

- **Backend**: FastAPI, WebSockets
- **Detección**: YOLOv8, TensorFlow Object Detection API
- **Streaming**: OpenCV, FFMPEG
- **Mapas**: Leaflet, Folium, Mapbox
- **Base de Datos**: PostgreSQL + PostGIS para datos geoespaciales

---

## 👤 Autor

**Desarrollado por Sebastian Yambay**  
Samsung Innovation Campus Ecuador 2025  
Repositorio: [Proyecto_Base_Conocimiento](https://github.com/SebastianYambay/Proyecto_Base_Conocimiento)

