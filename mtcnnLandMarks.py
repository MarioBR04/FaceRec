from mtcnn import MTCNN
import cv2
import os

# Cargar MTCNN
detector = MTCNN()

# Ruta a las imágenes
image_folder = 'input/lfw-dataset/lfw-deepfunneled'

# Crear un archivo de salida para los landmarks
landmarks_file = 'input/lfw-dataset/lfw_landmarks.lst'
with open(landmarks_file, 'w') as f:
    # Iterar sobre las carpetas (personas) en el directorio
    for person_folder in os.listdir(image_folder):
        person_path = os.path.join(image_folder, person_folder)

        # Verificar que es una carpeta
        if os.path.isdir(person_path):
            # Iterar sobre las imágenes dentro de cada carpeta
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)

                # Leer la imagen
                img = cv2.imread(img_path)

                # Verificar si la imagen se cargó correctamente
                if img is None:
                    print(f"Error al leer la imagen: {img_path}")
                    continue  # Saltar a la siguiente imagen si no se puede leer

                # Detectar los rostros y los landmarks
                results = detector.detect_faces(img)

                # Si se detectaron rostros
                if results:
                    face = results[0]  # Tomar el primer rostro encontrado
                    landmarks = face['keypoints']  # Diccionario con los 5 puntos clave

                    # Escribir los landmarks en el archivo
                    line = f"{person_folder}/{img_name} "
                    for key in landmarks:
                        line += f"{landmarks[key][0]} {landmarks[key][1]} "
                    f.write(line.strip() + '\n')
                else:
                    print(f"No se detectaron rostros en la imagen: {img_path}")