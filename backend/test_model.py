# test_simple.py
import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'inaniSite.settings')
import django
django.setup()

from asl_api.ml_utils import processor

# Crear una imagen de prueba (200x200 en negro)
test_frame = np.zeros((200, 200, 3), dtype=np.uint8)

# Probar una predicción
result = processor.predict_frame(test_frame)
print(f"Predicción de prueba: {result['letter']} (confianza: {result['confidence']:.2%})")