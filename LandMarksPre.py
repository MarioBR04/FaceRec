import os
import cv2
import numpy as np
import onnxruntime as ort
import insightface

print(ort.get_device())  # Debería imprimir "GPU"

# Cargar modelo con soporte para 68 landmarks
app = insightface.app.FaceAnalysis(
    name='buffalo_l',
    root="models/insightface",
    providers=['CUDAExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))

print("Modelo InsightFace cargado con éxito.")

# Ruta a las imágenes
image_folder = 'input/lfw-dataset/lfw-deepfunneled'
landmarks_file = 'input/lfw-dataset/lfw_landmarks.lst'

# Diccionario para mapear persona → class_id
person_to_id = {}

with open(landmarks_file, 'w') as f:
    idx = 0  # contador global de imágenes
    for person_folder in sorted(os.listdir(image_folder)):
        person_path = os.path.join(image_folder, person_folder)

        if os.path.isdir(person_path):
            # Asignar ID único por persona
            if person_folder not in person_to_id:
                person_to_id[person_folder] = len(person_to_id)
            class_id = person_to_id[person_folder]

            for img_name in sorted(os.listdir(person_path)):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Error al leer la imagen: {img_path}")
                    continue

                faces = app.get(img)

                if faces:
                    face = faces[0]  # Solo primer rostro

                    if hasattr(face, 'landmark_3d_68'):
                        landmark_array = face.landmark_3d_68  # (68, 3) ndarray
                        landmark_array = landmark_array[:, :2]  # Usamos solo (x, y)

                        if landmark_array.shape != (68, 2):
                            print(f"Landmarks mal formateados para {img_path}")
                            continue

                        # Ruta relativa
                        rel_path = f"{person_folder}/{img_name}"

                        # Formatear puntos como "x y"
                        landmarks_str = ' '.join(f"{int(x)} {int(y)}" for (x, y) in landmark_array)

                        # Escribir línea con índice, ruta, class_id y landmarks
                        f.write(f"{idx} {rel_path} {class_id} {landmarks_str}\n")
                        idx += 1
                        print(f"{rel_path} tiene {len(landmarks_str.split())} valores, class_id={class_id}")

                    else:
                        print(f"El rostro detectado no tiene 68 landmarks: {img_path}")
                else:
                    print(f"No se detectaron rostros en la imagen: {img_path}")
