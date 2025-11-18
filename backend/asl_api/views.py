# asl_api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .ml_utils import processor
import os
import cv2
import numpy as np

class UploadVideoView(APIView):
    def post(self, request):
        video = request.FILES.get('video')
        
        if not video:
            return Response(
                {
                    'success': False,
                    'error': 'No se envió ningún video'
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        allowed_formats = ['.mp4', '.mov', '.webm', '.avi']
        file_ext = os.path.splitext(video.name)[1].lower()
        
        if file_ext not in allowed_formats:
            return Response(
                {
                    'success': False,
                    'error': f'Formato no permitido. Use: {", ".join(allowed_formats)}'
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            video_path = default_storage.save(
                f'uploads/{video.name}', 
                ContentFile(video.read())
            )
            full_path = default_storage.path(video_path)
        except Exception as e:
            return Response(
                {
                    'success': False,
                    'error': f'Error al guardar el video: {str(e)}'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        try:
            # 🤖 PROCESAR VIDEO CON IA
            # confidence_threshold ajustable:
            # - 0.3 = Más permisivo (captura más predicciones, puede tener ruido)
            # - 0.5 = Balanceado
            # - 0.7 = Más estricto (solo predicciones muy confiables)
            result = processor.process_video(
                full_path, 
                confidence_threshold=0.3,
                save_debug_frames=True  # Guardar frames para debugging
            )
            
            # Obtener información del video
            video_size_mb = video.size / (1024 * 1024)
            
            # Opcional: Eliminar video temporal después de procesarlo
            # Descomenta la siguiente línea si quieres borrar el video después
            # os.remove(full_path)
            
            # Verificar si hubo errores en el procesamiento
            if 'error' in result:
                return Response(
                    {
                        'success': False,
                        'message': 'Video recibido pero hubo un error al procesarlo',
                        'video_info': {
                            'filename': video.name,
                            'size_mb': round(video_size_mb, 2),
                            'format': file_ext,
                        },
                        'error': result['error']
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # Respuesta exitosa
            return Response({
                'success': True,
                'message': '¡Video procesado correctamente! 🎥',
                'video_info': {
                    'filename': video.name,
                    'size_mb': round(video_size_mb, 2),
                    'format': file_ext,
                    'saved_path': video_path
                },
                'ai_results': {
                    'detected_word': result.get('word', ''),
                    'frames_analyzed': result.get('frames_analyzed', 0),
                    'total_predictions': result.get('total_predictions', 0),
                    'confidence_threshold': result.get('confidence_threshold', 0.3),
                    # Enviar solo las primeras 10 predicciones para no saturar la respuesta
                    'predictions': result.get('predictions', [])[:10],
                    # Ruta donde se guardaron los frames de debug
                    'debug_frames_path': result.get('debug_frames_path')
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            # Capturar cualquier error inesperado
            return Response(
                {
                    'success': False,
                    'error': f'Error inesperado al procesar el video: {str(e)}'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class UploadImageView(APIView):
    """
    Vista para subir y procesar imágenes de lenguaje de señas (ASL).
    
    POST /api/upload-image/
    Recibe una imagen y utiliza el modelo de IA para detectar la letra del alfabeto ASL.
    """
    
    def post(self, request):
        # Obtener la imagen del request
        image = request.FILES.get('image')
        
        # Validar que se envió una imagen
        if not image:
            return Response(
                {
                    'success': False,
                    'error': 'No se envió ninguna imagen'
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validar el formato de la imagen
        allowed_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        file_ext = os.path.splitext(image.name)[1].lower()
        
        if file_ext not in allowed_formats:
            return Response(
                {
                    'success': False,
                    'error': f'Formato no permitido. Use: {", ".join(allowed_formats)}'
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Leer imagen directamente en memoria (sin guardar en disco)
        try:
            file_data = np.frombuffer(image.read(), np.uint8)
        except Exception as e:
            return Response(
                {
                    'success': False,
                    'error': f'Error al leer la imagen: {str(e)}'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        try:
            # Decodificar imagen
            frame = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
            
            if frame is None:
                return Response(
                    {
                        'success': False,
                        'error': 'No se pudo decodificar la imagen'
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # 🤖 HACER PREDICCIÓN EN LA IMAGEN
            result = processor.predict_frame(frame)
            
            # Obtener información de la imagen
            image_size_mb = image.size / (1024 * 1024)
            height, width = frame.shape[:2]
            
            # Respuesta exitosa
            return Response({
                'success': True,
                'message': '¡Imagen procesada correctamente! 📸',
                'image_info': {
                    'filename': image.name,
                    'size_mb': round(image_size_mb, 2),
                    'format': file_ext,
                    'dimensions': f"{width}x{height}"
                },
                'ai_results': {
                    'detected_letter': result.get('letter', '?'),
                    'confidence': result.get('confidence', 0.0),
                    'confidence_percentage': f"{result.get('confidence', 0.0) * 100:.1f}%"
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            # Capturar cualquier error inesperado
            return Response(
                {
                    'success': False,
                    'error': f'Error inesperado al procesar la imagen: {str(e)}'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )