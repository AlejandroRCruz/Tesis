import cv2
import time
import csv
from ultralytics import YOLO
import numpy as np
import os
import psutil

# ---------------------------
# CONFIGURACIÓN
# ---------------------------
VIDEO_PATH = "C:/Users/Alejandro/Documents/GitHub/Tesis/recortes/Frente_5m.mp4"
OUTPUT_CSV = "estadisticasF5m_nano_Color.csv"
OUTPUT_TXT = "estadisticasF5m_nano_Color.txt"

model = YOLO("yolov8n-pose.pt")  # Ligero y rápido

# ---------------------------
# ABRIR VIDEO Y OBTENER RESOLUCIÓN
# ---------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ No se pudo abrir el video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

min_size = min(width, height)
scale_w = min_size / width
scale_h = min_size / height

print(f"Video original: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
print(f"Resolución ajustada a: {int(width*scale_w)}x{int(height*scale_h)}")

# ---------------------------
# PREPARAR CSV
# ---------------------------
csv_headers = ["frame",
               "dist_euclid_shlders", "dist_manh_shlders",
               "dist_euclid_left_shoulder_hip", "dist_manh_left_shoulder_hip",
               "dist_euclid_right_shoulder_hip", "dist_manh_right_shoulder_hip",
               "dist_euclid_hips", "dist_manh_hips",
               "time_frame_sec", "cpu_percent", "memory_percent"]

csv_file = open(OUTPUT_CSV, mode='w', newline='')
writer = csv.writer(csv_file)
writer.writerow(csv_headers)

# ---------------------------
# PROCESAR FRAMES
# ---------------------------
frame_num = 0
tiempo_total = 0
process = psutil.Process(os.getpid())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar y B/N antes de contar tiempo
    frame_resized = cv2.resize(frame, (int(width*scale_w), int(height*scale_h)))

    start_frame_time = time.time()  # Tiempo exacto de procesamiento

    # Detección de keypoints
    results = model(frame_resized, verbose=False)
    
    # Inicializar distancias
    dist_euclid_shlders = dist_manh_shlders = 0
    dist_euclid_lsh_hip = dist_manh_lsh_hip = 0
    dist_euclid_rsh_hip = dist_manh_rsh_hip = 0
    dist_euclid_hips = dist_manh_hips = 0

    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        if len(keypoints) > 0:
            person = keypoints[0]
            ls = person[5].astype(float)
            rs = person[6].astype(float)
            lh = person[11].astype(float)
            rh = person[12].astype(float)

            # Distancias Euclidiana
            dist_euclid_shlders = np.linalg.norm(ls - rs)
            dist_euclid_lsh_hip = np.linalg.norm(ls - lh)
            dist_euclid_rsh_hip = np.linalg.norm(rs - rh)
            dist_euclid_hips = np.linalg.norm(lh - rh)

            # Distancias Manhattan
            dist_manh_shlders = np.sum(np.abs(ls - rs))
            dist_manh_lsh_hip = np.sum(np.abs(ls - lh))
            dist_manh_rsh_hip = np.sum(np.abs(rs - rh))
            dist_manh_hips = np.sum(np.abs(lh - rh))

    tiempo_frame = time.time() - start_frame_time
    tiempo_total += tiempo_frame

    # Recursos medidos independientemente, no suman al tiempo
    cpu_percent = process.cpu_percent(interval=0.0)/psutil.cpu_count()
    memory_percent = process.memory_percent()

    # Guardar en CSV
    writer.writerow([frame_num,
                     dist_euclid_shlders, dist_manh_shlders,
                     dist_euclid_lsh_hip, dist_manh_lsh_hip,
                     dist_euclid_rsh_hip, dist_manh_rsh_hip,
                     dist_euclid_hips, dist_manh_hips,
                     tiempo_frame,
                     cpu_percent, memory_percent])

    # Mostrar frame
    cv2.imshow("Video B/N con Keypoints", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()
csv_file.close()

# ---------------------------
# ESTADÍSTICAS FINALES
# ---------------------------
tiempo_video_real = total_frames / fps
factor = tiempo_total / tiempo_video_real

estadisticas = f"""
Frames procesados: {frame_num}/{total_frames}
Duración del video real: {tiempo_video_real:.2f} seg
Tiempo total de procesamiento (solo inferencia y cálculo distancias): {tiempo_total:.2f} seg
Factor de procesamiento vs tiempo real: {factor:.2f}x
CSV guardado en: {OUTPUT_CSV}
"""

print(estadisticas)

# Guardar estadísticas en TXT
with open(OUTPUT_TXT, "w") as f:
    f.write(estadisticas)

print(f"📄 Estadísticas finales guardadas en: {OUTPUT_TXT}")
