# ASL Recognition Backend API
Backend API desarrollado con Django REST Framework para reconocimiento de lenguaje de señas americano (ASL) usando inteligencia artificial.

## Descripción

Este proyecto proporciona una API REST que permite:
- **Procesar videos** de señas ASL y detectar letras del alfabeto
- **Procesar imágenes** individuales de señas ASL
- Utilizar un modelo pre-entrenado basado en **Inception V3** con transfer learning
- Devolver predicciones con niveles de confianza

El modelo fue entrenado con el [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) disponible en Kaggle.

## 🏗️ Arquitectura

```
┌─────────────┐
│   Frontend  │ (React Native / Web)
│   Móvil/Web │
└──────┬──────┘
       │ HTTP POST (multipart/form-data)
       │ 
┌──────▼──────────────────────────────────┐
│        Django REST Framework            │
│  ┌────────────────────────────────┐    │
│  │  /api/upload-video/            │    │
│  │  - Recibe video MP4/MOV/WebM   │    │
│  │  - Extrae frames clave         │    │
│  │  - Procesa cada frame          │    │
│  └────────────┬───────────────────┘    │
│               │                         │
│  ┌────────────▼───────────────────┐    │
│  │  /api/upload-image/            │    │
│  │  - Recibe imagen JPG/PNG       │    │
│  │  - Procesa imagen única        │    │
│  └────────────┬───────────────────┘    │
└───────────────┼─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│      ASL Video Processor                │
│   ┌─────────────────────────────────┐  │
│   │  1. Extrae frames del video     │  │
│   │  2. Preprocesa cada frame       │  │
│   │  3. Mantiene aspect ratio       │  │
│   │  4. Redimensiona a 200x200      │  │
│   └───────────┬─────────────────────┘  │
│               │                         │
│   ┌───────────▼─────────────────────┐  │
│   │  Modelo TensorFlow (Inception)  │  │
│   │  - trained_model_graph.pb       │  │
│   │  - 29 clases (A-Z + space...)   │  │
│   └───────────┬─────────────────────┘  │
│               │                         │
│   ┌───────────▼─────────────────────┐  │
│   │  Agrupación de predicciones     │  │
│   │  [A,A,A,B,B] → "AB"             │  │
│   └─────────────────────────────────┘  │
└─────────────────────────────────────────┘
                │
        ┌───────▼──────┐
        │   Respuesta  │
        │     JSON     │
        └──────────────┘
```

## 🚀 Instalación

### 1. Requisitos previos

```bash
Python 3.8+
pip
virtualenv (recomendado)
```

### 2. Clonar el repositorio

```bash
git clone <tu-repositorio>
cd backend
```

### 3. Crear entorno virtual

```bash
python -m venv env

# Windows
env\Scripts\activate

# Linux/Mac
source env/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
django==4.2.0
djangorestframework==3.14.0
django-cors-headers==4.0.0
tensorflow==2.15.0
opencv-python==4.12.0
numpy==1.26.4
pillow==10.0.0
```

### 5. Descargar el modelo pre-entrenado

```bash
cd asl_api
mkdir ml_models
cd ml_models

# Descargar modelo (90MB)
curl -L "https://github.com/grassknoted/Unvoiced/raw/master/trained_model_graph.pb" -o trained_model_graph.pb

# Descargar etiquetas
curl -L "https://github.com/grassknoted/Unvoiced/raw/master/training_set_labels.txt" -o training_set_labels.txt
```

O descarga manualmente desde: https://github.com/grassknoted/Unvoiced

### 6. Configurar Django

```bash
# Volver al directorio raíz del proyecto
cd ../../

# Hacer migraciones
python manage.py makemigrations
python manage.py migrate

# Crear carpetas necesarias
mkdir -p media/uploads
```

### 7. Iniciar el servidor

```bash
python manage.py runserver
```

El servidor estará disponible en: `http://localhost:8000`

## 📡 Endpoints de la API

### 1. Procesar Video

**POST** `/api/upload-video/`

Procesa un video completo y detecta las letras de señas ASL.

**Request:**
```http
POST /api/upload-video/ HTTP/1.1
Content-Type: multipart/form-data

video: <archivo.mp4>
```

**Response (Success):**
```json
{
  "success": true,
  "message": "¡Video procesado correctamente! 🎥",
  "video_info": {
    "filename": "mi_video.mp4",
    "size_mb": 2.5,
    "format": ".mp4",
    "saved_path": "uploads/mi_video.mp4"
  },
  "ai_results": {
    "detected_word": "HELLO",
    "frames_analyzed": 15,
    "total_predictions": 12,
    "confidence_threshold": 0.15,
    "predictions": [
      {"letter": "H", "confidence": 0.85},
      {"letter": "E", "confidence": 0.78},
      {"letter": "L", "confidence": 0.82},
      {"letter": "L", "confidence": 0.79},
      {"letter": "O", "confidence": 0.76}
    ],
    "debug_frames_path": "/path/to/debug_frames"
  }
}
```

**Formatos soportados:** `.mp4`, `.mov`, `.webm`, `.avi`

---

### 2. Procesar Imagen

**POST** `/api/upload-image/`

Procesa una imagen única y detecta la letra de seña ASL.

**Request:**
```http
POST /api/upload-image/ HTTP/1.1
Content-Type: multipart/form-data

image: <archivo.jpg>
```

**Response (Success):**
```json
{
  "success": true,
  "message": "¡Imagen procesada correctamente! 📸",
  "image_info": {
    "filename": "seña_A.jpg",
    "size_mb": 0.8,
    "format": ".jpg",
    "dimensions": "640x480"
  },
  "ai_results": {
    "detected_letter": "A",
    "confidence": 0.92,
    "confidence_percentage": "92.0%"
  }
}
```

**Formatos soportados:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

---

### Error Response

```json
{
  "success": false,
  "error": "Descripción del error"
}
```

## 🎛️ Configuración

### Ajustar umbral de confianza

En `asl_api/views.py`, línea ~54:

```python
result = processor.process_video(
    full_path, 
    confidence_threshold=0.15,  # Cambiar este valor (0.0 - 1.0)
    save_debug_frames=True
)
```

**Valores recomendados:**
- `0.15` (15%) - Más permisivo, captura más predicciones
- `0.30` (30%) - Balanceado
- `0.50` (50%) - Más estricto, solo predicciones muy confiables

### Número de frames a extraer

En `asl_api/ml_utils.py`, método `extract_frames`:

```python
frames = self.extract_frames(
    video_path, 
    max_frames=15,    # Máximo de frames a procesar
    skip_frames=5     # Procesar cada N frames
)
```

### Debug frames

Para ver qué frames está procesando el modelo:

```python
result = processor.process_video(
    full_path, 
    save_debug_frames=True  # Guardar frames en asl_api/debug_frames/
)
```

Los frames se guardan en `asl_api/debug_frames/`:
- `original_XXX.jpg` - Frame original del video
- `processed_XXX.jpg` - Frame después del preprocesamiento

## 🧪 Testing

### Probar con cURL

**Video:**
```bash
curl -X POST http://localhost:8000/api/upload-video/ \
  -F "video=@/path/to/video.mp4"
```

**Imagen:**
```bash
curl -X POST http://localhost:8000/api/upload-image/ \
  -F "image=@/path/to/image.jpg"
```

### Mock Frontend (HTML)

Incluye archivos HTML de prueba en el repositorio:
- `test_upload_video.html` - Para probar upload de videos
- `test_upload_image.html` - Para probar upload de imágenes

Simplemente abre estos archivos en tu navegador y sube archivos.

## 📁 Estructura del Proyecto

```
backend/
├── manage.py
├── requirements.txt
├── README.md
├── test_upload_video.html          # Mock frontend para videos
├── test_upload_image.html          # Mock frontend para imágenes
│
├── inaniSite/                       # Configuración Django
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── asl_api/                         # App principal
│   ├── __init__.py
│   ├── views.py                     # Endpoints REST
│   ├── urls.py
│   ├── ml_utils.py                  # Lógica del modelo de IA
│   │
│   ├── ml_models/                   # Modelo pre-entrenado
│   │   ├── trained_model_graph.pb   # Modelo TensorFlow (90MB)
│   │   └── training_set_labels.txt  # Etiquetas A-Z
│   │
│   └── debug_frames/                # Frames guardados para debug
│       ├── original_000.jpg
│       ├── processed_000.jpg
│       └── ...
│
└── media/
    └── uploads/                     # Videos/imágenes subidos
```

## 🎥 Mejores Prácticas para Videos

Para obtener mejores resultados:

1. **Orientación:** Graba en **horizontal** (landscape), no vertical
2. **Fondo:** Usa un fondo **claro y uniforme** (blanco ideal)
3. **Iluminación:** Buena iluminación frontal, evita sombras
4. **Centrado:** Mantén la mano **centrada** en el frame
5. **Estabilidad:** Mantén la seña **estática** 2-3 segundos
6. **Claridad:** Dedos bien separados y visibles
7. **Duración:** Videos cortos (2-5 segundos) funcionan mejor

### ❌ Evitar:
- Videos verticales muy estrechos
- Fondos con muchos objetos o patrones
- Mala iluminación o contraluces
- Manos fuera del centro
- Movimiento rápido o borroso

## 🐛 Troubleshooting

### Error: "Broken pipe"
**Causa:** El procesamiento toma mucho tiempo.  
**Solución:** Reduce `max_frames` o `skip_frames` en `ml_utils.py`

### Error: "No se pudieron extraer frames"
**Causa:** Formato de video no compatible o archivo corrupto.  
**Solución:** Convierte el video a MP4 con codec H.264

### Error: "AttributeError: debug_frames_dir"
**Causa:** Falta inicializar en `__init__`.  
**Solución:** Verifica que `debug_frames_dir` esté en el método `__init__`

### Confianza muy baja en todas las predicciones
**Causa:** Preprocesamiento distorsiona la imagen.  
**Solución:** Verifica que `preprocess_frame` use padding en lugar de resize directo

### Error: "numpy.core._multiarray_umath failed to import"
**Causa:** Incompatibilidad entre TensorFlow y NumPy 2.x.  
**Solución:** 
```bash
pip uninstall numpy
pip install numpy==1.26.4
```

## 🔐 Seguridad

**⚠️ IMPORTANTE:** Este código es solo para desarrollo/demo.

Para producción, implementa:
- ✅ Autenticación JWT/OAuth
- ✅ Rate limiting
- ✅ Validación de tamaño de archivos
- ✅ Escaneo de virus en uploads
- ✅ HTTPS obligatorio
- ✅ Variables de entorno para configuración
- ✅ Logging y monitoreo

## 📊 Rendimiento

**Tiempos aproximados:**
- Carga del modelo: ~3 segundos (primera vez)
- Predicción por imagen: ~0.2 segundos
- Procesamiento de video (15 frames): ~5-8 segundos

**Requisitos de hardware:**
- CPU: Funciona bien con CPU moderna (Intel i5+)
- RAM: Mínimo 4GB, recomendado 8GB
- GPU: Opcional (TensorFlow puede usar CUDA si está disponible)

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -am 'Agrega nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

## 📄 Licencia

Este proyecto usa el modelo de [Unvoiced](https://github.com/grassknoted/Unvoiced) que está disponible públicamente.

## 👥 Créditos

- **Modelo original:** [grassknoted/Unvoiced](https://github.com/grassknoted/Unvoiced)
- **Dataset:** [ASL Alphabet Dataset en Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Framework:** Django REST Framework
- **ML:** TensorFlow / Inception V3

## 📞 Soporte

¿Tienes problemas? Abre un issue en GitHub o contacta al equipo de desarrollo.

---

**Desarrollado con ❤️ para mejorar la accesibilidad** 🤟