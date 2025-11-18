import cv2
import numpy as np
import os
import tensorflow as tf
from collections import Counter
import time

class ASLVideoProcessor:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'trained_model_graph.pb')
        labels_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'training_set_labels.txt')
        
        # Crear carpeta para debug de frames
        self.debug_frames_dir = os.path.join(os.path.dirname(__file__), 'debug_frames')
        if not os.path.exists(self.debug_frames_dir):
            os.makedirs(self.debug_frames_dir)
            print(f"📁 Carpeta de debug creada: {self.debug_frames_dir}")
        
        # Cargar las etiquetas
        try:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            print(f"✅ Etiquetas cargadas: {len(self.labels)} clases")
        except Exception as e:
            print(f"⚠️ Error al cargar etiquetas: {e}")
            self.labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # Cargar el modelo de TensorFlow 1.x
        try:
            # Deshabilitar warnings de TensorFlow
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            # Cargar el grafo del modelo
            with tf.io.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            
            # Crear una sesión de TensorFlow
            self.graph = tf.Graph()
            with self.graph.as_default():
                tf.import_graph_def(graph_def, name='')
            
            self.sess = tf.compat.v1.Session(graph=self.graph)
            self.softmax_tensor = self.graph.get_tensor_by_name('final_result:0')
            
            print("✅ Modelo cargado correctamente")
            
        except Exception as e:
            print(f"⚠️ Error al cargar el modelo: {e}")
            self.sess = None
            self.softmax_tensor = None
    
    def extract_frames(self, video_path, max_frames=30, skip_frames=3):
        """
        Extrae frames del video.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("No se pudo abrir el video")
        
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % skip_frames == 0:
                frames.append(frame)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"✅ Extraídos {len(frames)} frames del video")
        return frames
    
    def extract_frames_from_capture(self, capture, max_frames=15, skip_frames=5):
        """
        Extrae frames desde cv2.VideoCapture igual que extract_frames,
        pero sin usar un archivo de video.
        """
        frames = []
        frame_count = 0

        while len(frames) < max_frames:
            ret, frame = capture.read()
            if not ret:
                print(f"⚠️ No se pudo leer frame {frame_count}")
                break

            if frame_count % skip_frames == 0:
                frames.append(frame)
                print(f"✅ Frame {len(frames)} extraído")

            frame_count += 1
        
        print(f"📊 Total de frames extraídos: {len(frames)}")
        return frames  # ✅ AGREGADO: Retornar los frames
    
    def preprocess_frame(self, frame):
        """
        Preprocesar frame MANTENIENDO el aspect ratio.
        Esto es CRÍTICO para que el modelo funcione correctamente.
        """
        height, width = frame.shape[:2]
        
        print(f"    📐 Original: {width}x{height}", end=' ')
        
        # Paso 1: Hacer la imagen cuadrada SIN distorsionar
        if height > width:
            # Imagen vertical - agregar padding a los lados
            diff = height - width
            pad_left = diff // 2
            pad_right = diff - pad_left
            frame = cv2.copyMakeBorder(
                frame, 
                0, 0,  # top, bottom
                pad_left, pad_right,  # left, right
                cv2.BORDER_CONSTANT, 
                value=[0, 0, 0]  # padding negro
            )
            print(f"→ Padding lateral: {height}x{height}", end=' ')
        elif width > height:
            # Imagen horizontal - agregar padding arriba y abajo
            diff = width - height
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            frame = cv2.copyMakeBorder(
                frame,
                pad_top, pad_bottom,  # top, bottom
                0, 0,  # left, right
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]  # padding negro
            )
            print(f"→ Padding vertical: {width}x{width}", end=' ')
        else:
            print(f"→ Ya es cuadrada", end=' ')
        
        # Paso 2: Redimensionar a 200x200 (ahora SÍ mantiene proporciones)
        resized_frame = cv2.resize(frame, (200, 200), interpolation=cv2.INTER_AREA)
        print(f"→ Redimensionada: 200x200")
        
        return resized_frame

    def predict_frame(self, frame, save_debug=False, frame_index=0):
        """
        Predecir la letra en un frame usando el modelo de Unvoiced.
        """
        start = time.time()
        if self.sess is None or self.softmax_tensor is None:
            return {'letter': '?', 'confidence': 0.0}
        
        try:
            processed_frame = self.preprocess_frame(frame)
            
            # 🔍 GUARDAR FRAME PARA DEBUG
            if save_debug:
                debug_dir = os.path.join(os.path.dirname(__file__), 'debug_frames')
                os.makedirs(debug_dir, exist_ok=True)
                
                # Guardar frame original
                original_path = os.path.join(debug_dir, f'frame_{frame_index:03d}_original.jpg')
                cv2.imwrite(original_path, frame)
                
                # Guardar frame procesado (el que ve el modelo)
                processed_path = os.path.join(debug_dir, f'frame_{frame_index:03d}_processed.jpg')
                cv2.imwrite(processed_path, processed_frame)
                
                print(f"💾 Frames guardados en {debug_dir}/")
            
            # Codificar frame como JPEG (igual que el código original)
            _, encoded_image = cv2.imencode('.jpg', processed_frame)
            image_data = encoded_image.tobytes()
            
            # Hacer predicción
            predictions = self.sess.run(
                self.softmax_tensor,
                {'DecodeJpeg/contents:0': image_data}
            )

            elapsed = time.time() - start
            
            # Obtener top 3 predicciones para debug
            top_k = predictions[0].argsort()[-5:][::-1]
            
            max_score = 0.0
            predicted_letter = '?'
            top_predictions = []
            
            for node_id in top_k[:5]:  # Top 5 para debug
                if node_id < len(self.labels):
                    human_string = self.labels[node_id]
                    score = float(predictions[0][node_id])
                    top_predictions.append(f"{human_string}:{score:.3f}")
                    
                    if score > max_score:
                        max_score = score
                        predicted_letter = human_string.upper()
            
            print(f"⏱️ {elapsed:.3f}s | Top: {', '.join(top_predictions)}")
            
            return {
                'letter': predicted_letter,
                'confidence': max_score
            }
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return {'letter': '?', 'confidence': 0.0}
    
    def process_video(self, video_path, confidence_threshold=0.4):
        try:
            print(f"🎬 Iniciando procesamiento de: {video_path}")
            frames = self.extract_frames(video_path, max_frames=30, skip_frames=5)
            
            if not frames:
                return {
                    'word': '',
                    'frames_analyzed': 0,
                    'predictions': [],
                    'total_predictions': 0,
                    'error': 'No se pudieron extraer frames del video'
                }
            
            print(f"🔍 Procesando {len(frames)} frames...")
            
            all_predictions = []
            
            for i, frame in enumerate(frames):
                print(f"  Frame {i+1}/{len(frames)}...", end=' ')
                prediction = self.predict_frame(frame)
                
                # Solo considerar predicciones con suficiente confianza
                if prediction['confidence'] >= confidence_threshold:
                    all_predictions.append(prediction)
                    print(f"✓ {prediction['letter']} ({prediction['confidence']:.2f})")
                else:
                    print(f"✗ Baja confianza ({prediction['confidence']:.2f})")
            
            print(f"✅ Procesamiento completado. {len(all_predictions)} predicciones válidas")
            
            # Agrupar predicciones para formar palabra
            word = self._group_predictions(all_predictions)
            
            return {
                'word': word,
                'frames_analyzed': len(frames),
                'predictions': all_predictions,
                'total_predictions': len(all_predictions),
                'confidence_threshold': confidence_threshold
            }
            
        except Exception as e:
            print(f"❌ Error en process_video: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'word': '',
                'frames_analyzed': 0,
                'predictions': [],
                'total_predictions': 0,
                'error': str(e)
            }

    def process_video_capture(self, capture, confidence_threshold=0.5):
        """
        Procesa un stream de VideoCapture (ej. cámara web) y devuelve las letras detectadas.
        """
        try:
            print("🎥 Iniciando procesamiento desde captura...")

            frames = self.extract_frames_from_capture(capture, max_frames=15, skip_frames=5)

            print(f'✅ Continuando con {len(frames) if frames else 0} frames')

            if not frames:
                return {
                    'word': '',
                    'frames_analyzed': 0,
                    'predictions': [],
                    'total_predictions': 0,
                    'error': 'No se pudieron extraer frames de la captura'
                }

            print(f"🔍 Procesando {len(frames)} frames capturados...")

            all_predictions = []

            for i, frame in enumerate(frames):
                print(f"  Frame {i+1}/{len(frames)}...", end=' ')
                prediction = self.predict_frame(frame)

                if prediction['confidence'] >= confidence_threshold:
                    all_predictions.append(prediction)
                    print(f"✓ {prediction['letter']} ({prediction['confidence']:.2f})")
                else:
                    print(f"✗ Baja confianza ({prediction['confidence']:.2f})")

            print(f"✅ Procesamiento finalizado. {len(all_predictions)} predicciones válidas")

            # Agrupar predicciones para formar la palabra
            word = self._group_predictions(all_predictions)

            return {
                'word': word,
                'frames_analyzed': len(frames),
                'predictions': all_predictions,
                'total_predictions': len(all_predictions),
                'confidence_threshold': confidence_threshold
            }

        except Exception as e:
            print(f"❌ Error en process_video_capture: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'word': '',
                'frames_analyzed': 0,
                'predictions': [],
                'total_predictions': 0,
                'error': str(e)
            }

    def _group_predictions(self, predictions):
        """
        Agrupa predicciones consecutivas para formar una palabra.
        Por ejemplo: [A, A, A, B, B, C, C, C] -> "ABC"
        """
        if not predictions:
            return ""
        
        word = ""
        prev_letter = None
        
        for pred in predictions:
            letter = pred['letter']
            if letter != prev_letter and letter != '?':
                word += letter
                prev_letter = letter
        
        return word

    def save_debug_frame(self, frame, index, prefix="frame"):
        """
        Guarda un frame para debugging.
        """
        try:
            filename = f"{prefix}_{index:03d}.jpg"
            filepath = os.path.join(self.debug_frames_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"  💾 Frame guardado: {filename}")
        except Exception as e:
            print(f"  ⚠️ No se pudo guardar frame: {e}")
    
    def process_video(self, video_path, confidence_threshold=0.3, save_debug_frames=True):
        """
        Procesa un video completo y devuelve las letras detectadas.
        
        Args:
            video_path: Ruta del video
            confidence_threshold: Umbral mínimo de confianza
            save_debug_frames: Si True, guarda los frames extraídos para debugging
        """
        try:
            print(f"🎬 Iniciando procesamiento de: {video_path}")
            
            # Extraer frames
            frames = self.extract_frames(video_path, max_frames=15, skip_frames=5)
            
            if not frames:
                return {
                    'word': '',
                    'frames_analyzed': 0,
                    'predictions': [],
                    'total_predictions': 0,
                    'error': 'No se pudieron extraer frames del video'
                }
            
            print(f"🔍 Procesando {len(frames)} frames...")
            
            # Predecir cada frame
            all_predictions = []
            
            for i, frame in enumerate(frames):
                # GUARDAR FRAME ORIGINAL para debugging
                if save_debug_frames:
                    self.save_debug_frame(frame, i, prefix="original")
                
                # GUARDAR FRAME PREPROCESADO para ver qué recibe el modelo
                if save_debug_frames:
                    processed = self.preprocess_frame(frame)
                    self.save_debug_frame(processed, i, prefix="processed")
                
                print(f"  Frame {i+1}/{len(frames)}...", end=' ')
                start_time = time.time()
                prediction = self.predict_frame(frame)
                elapsed = time.time() - start_time
                
                print(f"⏱️ {elapsed:.3f}s → {prediction['letter']} ({prediction['confidence']:.2f})", end='')
                
                # Solo considerar predicciones con suficiente confianza
                if prediction['confidence'] >= confidence_threshold:
                    all_predictions.append(prediction)
                    print(f" ✓ Aceptada")
                else:
                    print(f" ✗ Rechazada (< {confidence_threshold:.0%})")
            
            print(f"✅ Procesamiento completado. {len(all_predictions)} predicciones válidas")
            
            # Agrupar predicciones para formar palabra
            word = self._group_predictions(all_predictions)
            
            # Mostrar ruta de frames guardados
            if save_debug_frames:
                print(f"📁 Frames guardados en: {self.debug_frames_dir}")
            
            return {
                'word': word,
                'frames_analyzed': len(frames),
                'predictions': all_predictions,
                'total_predictions': len(all_predictions),
                'confidence_threshold': confidence_threshold,
                'debug_frames_path': self.debug_frames_dir if save_debug_frames else None
            }
            
        except Exception as e:
            print(f"❌ Error en process_video: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'word': '',
                'frames_analyzed': 0,
                'predictions': [],
                'total_predictions': 0,
                'error': str(e)
            }

    def __del__(self):
        """
        Cerrar la sesión de TensorFlow al destruir el objeto.
        """
        if hasattr(self, 'sess') and self.sess is not None:
            self.sess.close()
            print("🔒 Sesión de TensorFlow cerrada")

# Instancia global del procesador
processor = ASLVideoProcessor()