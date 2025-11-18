# test_video_real.py
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'inaniSite.settings')
import django
django.setup()

from asl_api.ml_utils import processor

# CAMBIA ESTO por el nombre de tu video
video_filename = 'test_03.mp4'
video_path = f'media/uploads/{video_filename}'

print(f"Procesando video: {video_path}")
result = processor.process_video(video_path)

print(f"\n🎯 Palabra detectada: {result.get('word', 'N/A')}")
print(f"📊 Frames analizados: {result.get('frames_analyzed', 0)}")
print(f"✅ Predicciones válidas: {result.get('total_predictions', 0)}")

if 'error' in result:
    print(f"❌ Error: {result['error']}")

if result.get('predictions'):
    print("\n📝 Predicciones detalladas:")
    for i, pred in enumerate(result['predictions'][:20]):
        print(f"  {i+1}. {pred['letter']} - Confianza: {pred['confidence']:.2%}")