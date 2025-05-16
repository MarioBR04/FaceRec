import cv2
import numpy as np
import time
import os
import mxnet as mx
from skimage import transform as trans

# === CONFIGURACIÓN DE MXNET Y MODELO ===
prefix = 'models/pretrain/model-softmax'
epoch = 20
SIMILARITY_THRESHOLD = 0.6

# === CONSTANTES PARA ALINEACIÓN FACIAL ===
# Landmarks estándar para rostros alineados de 112x112
src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

# Cargar contexto (GPU si está disponible, CPU en caso contrario)
ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
print(f"Contexto MXNet: {ctx}")

# === CARGA DE MODELO MXNET ===
try:
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))])
    mod.set_params(arg_params, aux_params)
    print("Modelo MXNet cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo MXNet: {str(e)}")
    print("Asegúrate de que el modelo existe en la ruta: models/pretrain/model-softmax-0020.params")
    exit(1)

# Intentar cargar detector de landmarks
try:
    face_landmark_detector = cv2.face.createFacemarkLBF()
    face_landmark_detector.loadModel("models/lbfmodel.yaml")
    landmark_detector_available = True
    print("Detector de landmarks faciales cargado correctamente")
except Exception as e:
    print(f"Error al cargar detector de landmarks: {e}")
    landmark_detector_available = False
    print("El detector de landmarks no está disponible. Se usará preprocesamiento básico.")

# === CONFIGURACIÓN YUNET ===
# Verificar si OpenCV tiene el módulo YuNet disponible
try:
    # Intentar cargar el detector YuNet
    face_detector = cv2.FaceDetectorYN_create(
        model="face_detection_yunet_2023mar.onnx",  # Intentar cargar el modelo preinstalado
        config="",
        input_size=(320, 320),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )
    yunet_available = True
    print("Detector YuNet cargado correctamente.")
except Exception as e:
    print(f"Error al cargar YuNet: {e}")
    print("Vamos a descargar el modelo YuNet...")
    yunet_available = False

# Si YuNet no está disponible, descargarlo
if not yunet_available:
    try:
        # Crear directorio para modelos si no existe
        if not os.path.exists("models"):
            os.makedirs("models")

        # URL del modelo YuNet
        model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        model_path = "models/face_detection_yunet_2023mar.onnx"

        # Descargar el modelo usando urllib
        import urllib.request

        print(f"Descargando modelo YuNet desde {model_url}...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Modelo guardado en {model_path}")

        # Cargar el detector YuNet con el modelo descargado
        face_detector = cv2.FaceDetectorYN_create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000
        )
        yunet_available = True
        print("Detector YuNet cargado correctamente desde el modelo descargado.")
    except Exception as e:
        print(f"Error al descargar/cargar YuNet: {e}")
        print("Fallback a HaarCascade...")
        # Utilizar HaarCascade como fallback
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_detector.empty():
            print("Error: No se pudo cargar ningún detector facial.")
            exit(1)
        else:
            print("HaarCascade cargado como alternativa.")
            yunet_available = False


# === PREPROCESAMIENTO DE IMÁGENES PARA MXNET ===
def preprocess_face_for_mxnet(face_img, landmarks=None):
    """Preprocesa una imagen facial para el modelo MXNet con alineación opcional de landmarks"""
    # Asegurarse de que la imagen esté en color (si es en escala de grises, convertir a RGB)
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    elif face_img.shape[2] == 1:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)

    # Si hay landmarks disponibles, alinear la cara
    if landmarks is not None and len(landmarks) == 5:
        dst = src.copy()
        tform = trans.SimilarityTransform()
        tform.estimate(landmarks, dst)
        M = tform.params[0:2, :]
        face_img = cv2.warpAffine(face_img, M, (112, 112), borderValue=0.0)
    else:
        # Si no hay landmarks, simplemente redimensionar
        face_img = cv2.resize(face_img, (112, 112))

    # Preprocesamiento para MobileNet:
    # 1. Convertir de BGR a RGB
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # 2. Normalizar valores a [0, 1]
    face_img = face_img.astype(np.float32) / 255.0
    # 3. Estandarizar según los valores de ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    face_img = (face_img - mean) / std
    # 4. Reorganizar dimensiones de (H,W,C) a (N,C,H,W)
    face_img = np.transpose(face_img, (2, 0, 1))
    face_img = np.expand_dims(face_img, axis=0)

    return face_img


# === DETECTOR DE LANDMARKS FACIALES ===
def detect_landmarks(face_img):
    """Detecta 5 landmarks faciales para alineación"""
    if not landmark_detector_available:
        return None

    # Convertir a escala de grises si es necesario
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img.copy()

    # El detector de landmarks puede requerir detectar caras antes
    faces = np.array([[0, 0, face_img.shape[1], face_img.shape[0]]])
    try:
        success, landmarks = face_landmark_detector.fit(gray, faces)
        if success:
            # Extraer los 5 landmarks principales (ojos, nariz, esquinas de la boca)
            # Esta parte puede necesitar ajustes según el detector de landmarks utilizado
            lmk = landmarks[0][0]
            if len(lmk) >= 68:  # Si es un detector de 68 puntos
                # Convertir 68 puntos a 5 puntos (estándar para MobileFaceNet)
                five_landmarks = np.array([
                    lmk[36],  # Ojo izquierdo esquina
                    lmk[45],  # Ojo derecho esquina
                    lmk[30],  # Nariz
                    lmk[48],  # Esquina izquierda de la boca
                    lmk[54]  # Esquina derecha de la boca
                ], dtype=np.float32)
                return five_landmarks
    except Exception as e:
        print(f"Error detectando landmarks: {e}")

    return None


# === EXTRACCIÓN DE CARACTERÍSTICAS CON MXNET ===
def extract_features(face_img):
    """Extrae características faciales usando el modelo MXNet cargado"""
    try:
        # Detectar landmarks si está disponible
        landmarks = detect_landmarks(face_img)

        # Preprocesar imagen
        preprocessed_face = preprocess_face_for_mxnet(face_img, landmarks)

        # Convertir a formato MXNet
        data = mx.nd.array(preprocessed_face)
        db = mx.io.DataBatch(data=(data,))

        # Forward pass a través del modelo
        mod.forward(db, is_train=False)

        # Obtener embeddings
        features = mod.get_outputs()[0].asnumpy()

        # Normalizar el vector de características (L2 normalization)
        features_norm = np.sqrt(np.sum(features ** 2, axis=1, keepdims=True))
        features = features / features_norm

        return features.flatten()
    except Exception as e:
        print(f"Error extrayendo características: {e}")
        return None


# === CÁLCULO DE SIMILITUD USANDO COSENO ===
def calculate_similarity(features1, features2):
    """Calcula la similitud de coseno entre dos vectores de características"""
    if features1 is None or features2 is None:
        return 0

    # Asegurarse de que los vectores son unidimensionales
    if len(features1.shape) > 1:
        features1 = features1.flatten()
    if len(features2.shape) > 1:
        features2 = features2.flatten()

    # Normalizar los vectores (por seguridad)
    features1 = features1 / np.linalg.norm(features1)
    features2 = features2 / np.linalg.norm(features2)

    # Similitud del coseno
    similarity = np.dot(features1, features2)

    # La similitud de coseno está entre -1 y 1, donde 1 es idéntico
    # Convertir a un valor entre 0 y 1
    similarity = (similarity + 1) / 2

    return similarity


# === CAPTURA DE CÁMARA ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara. Verifique la conexión o los permisos.")
    exit(1)

# Obtener dimensiones del frame
ret, frame = cap.read()
if not ret:
    print("Error al capturar imagen desde la cámara")
    exit(1)

frame_height, frame_width, _ = frame.shape

# Configurar el detector YuNet con el tamaño del frame
if yunet_available:
    face_detector.setInputSize((frame_width, frame_height))

# === FASE 1: CAPTURA DE ROSTRO DE REFERENCIA ===
print("=== CAPTURA DE ROSTRO DE REFERENCIA ===")
print("Presiona 's' para capturar un rostro como referencia")
print("Presiona 'ESC' para salir")

reference_face = None
reference_features = None
reference_name = input("Introduce un nombre para la persona de referencia: ")
if not reference_name:
    reference_name = "Persona"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar imagen desde la cámara")
        break

    # Crear una copia del frame para dibujar
    display_frame = frame.copy()

    # Detectar rostros con el método adecuado
    faces_detected = []
    if yunet_available:
        # Usar YuNet
        _, faces = face_detector.detect(frame)
        if faces is not None:
            for face in faces:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                confidence = face[14]
                if confidence > 0.9:  # Filtrar por confianza
                    faces_detected.append((x, y, w, h))
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # Usar HaarCascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        faces_detected = [(x, y, w, h) for (x, y, w, h) in faces]
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar instrucciones en pantalla
    cv2.putText(display_frame, "Presiona 's' para capturar", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Rostros detectados: {len(faces_detected)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Nombre: {reference_name}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar el frame
    try:
        cv2.imshow("Captura de rostro de referencia", display_frame)
    except Exception as e:
        print(f"Error al mostrar ventana: {e}")
        print("Ejecute el script de instalación de dependencias OpenCV primero.")
        break

    # Capturar tecla
    key = cv2.waitKey(1)

    # Si se presiona 's' y hay rostros detectados
    if key == ord('s') and len(faces_detected) > 0:
        # Tomar el primer rostro como referencia
        x, y, w, h = faces_detected[0]
        reference_face = frame[y:y + h, x:x + w].copy()

        try:
            # Extraer características con MXNet
            reference_features = extract_features(reference_face)

            if reference_features is None:
                print("❌ Error al extraer características del rostro. Intenta de nuevo.")
                continue

            # Verificar que el vector de características es válido
            if np.isnan(reference_features).any():
                print("❌ Vector de características contiene NaN. Intenta de nuevo.")
                continue

            # Guardar la imagen de referencia
            if not os.path.exists("faces"):
                os.makedirs("faces")
            cv2.imwrite(f"faces/reference_{reference_name}.jpg", reference_face)

            print(f"✅ Rostro de referencia de {reference_name} capturado y guardado")
            print(f"Vector de características shape: {reference_features.shape}")
            print(f"Norma del vector: {np.linalg.norm(reference_features)}")
            time.sleep(1)  # Pausa breve para confirmación
            break
        except Exception as e:
            print(f"Error al procesar el rostro de referencia: {e}")
            continue

    elif key == 27:  # ESC
        print("Programa terminado por el usuario.")
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

# === FASE 2: RECONOCIMIENTO EN TIEMPO REAL ===
print("\n=== RECONOCIMIENTO EN TIEMPO REAL ===")
print("Presiona 'ESC' para salir")

# Usar el umbral de similitud definido al inicio
similarity_threshold = SIMILARITY_THRESHOLD

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar imagen desde la cámara")
        break

    # Crear una copia del frame para dibujar
    display_frame = frame.copy()

    # Detectar rostros con el método adecuado
    faces_detected = []
    if yunet_available:
        # Usar YuNet
        _, faces = face_detector.detect(frame)
        if faces is not None:
            for face in faces:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                confidence = face[14]
                if confidence > 0.9:  # Filtrar por confianza
                    faces_detected.append((x, y, w, h))
    else:
        # Usar HaarCascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        faces_detected = [(x, y, w, h) for (x, y, w, h) in faces]

    # Procesar cada rostro detectado
    for (x, y, w, h) in faces_detected:
        try:
            # Extraer región del rostro
            face_roi = frame[y:y + h, x:x + w]

            # Extraer características con MXNet
            face_features = extract_features(face_roi)

            # Calcular similitud con el rostro de referencia
            if reference_features is not None and face_features is not None:
                similarity = calculate_similarity(face_features, reference_features)

                # Determinar si es una coincidencia
                is_match = similarity > similarity_threshold
                label = "MATCH" if is_match else "NO MATCH"
                color = (0, 255, 0) if is_match else (0, 0, 255)

                # Dibujar rectángulo y etiqueta
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, f"{label} ({similarity:.2f})",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception as e:
            # Si hay error al procesar, simplemente dibujamos un rectángulo amarillo
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(display_frame, f"Error: {str(e)[:10]}...", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Mostrar umbrales e instrucciones en pantalla
    cv2.putText(display_frame, f"Umbral: {similarity_threshold}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, "Presiona ESC para salir", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar el frame
    try:
        cv2.imshow("Reconocimiento facial con MXNet", display_frame)
    except Exception as e:
        print(f"Error al mostrar ventana: {e}")
        break

    # Salir si se presiona ESC
    if cv2.waitKey(1) == 27:
        print("Programa terminado por el usuario.")
        break

print("Proceso completado.")
cap.release()
cv2.destroyAllWindows()