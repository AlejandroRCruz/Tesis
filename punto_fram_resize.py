import cv2
import time
import csv
from ultralytics import YOLO

# Cargar modelo pose nano
model = YOLO("yolov8n-pose.pt")  #

# Video de entrada
cap = cv2.VideoCapture("C:/Users/Alejandro/Documents/GitHub/Tesis/Videos Necesarios/prueba 3.mp4")

# Archivo CSV de salida
csv_file = open("resultados_pose_img_resized.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame_id", "tiempo_ms", "persona_id", "lado",
    "x_hombro", "y_hombro", "conf_hombro",
    "x_cadera", "y_cadera", "conf_cadera"
])

# IDs COCO keypoints
pair_left = [5, 11]   # hombro izq, cadera izq
pair_right = [6, 12]  # hombro der, cadera der

# Resolución de entrada manual
target_w, target_h = 640, 384

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ----------- Resize manual del frame (manteniendo proporción) -----------
    frame_resized = cv2.resize(frame, (target_w, target_h))

    # ----------- INICIO cronómetro -----------
    t0 = time.time()
    results = model(frame_resized, conf=0.6, iou=0.45, classes=[0], max_det=20)
    personas_detectadas = []

    for r in results:
        if r.keypoints is None:
            continue

        kps = r.keypoints.xy.cpu().numpy()
        confs = r.keypoints.conf.cpu().numpy()

        for pid, (person_kp, person_conf) in enumerate(zip(kps, confs)):
            left_conf = (person_conf[pair_left[0]] + person_conf[pair_left[1]]) / 2
            right_conf = (person_conf[pair_right[0]] + person_conf[pair_right[1]]) / 2

            if left_conf >= right_conf:
                selected = pair_left
                lado = "Izq"
            else:
                selected = pair_right
                lado = "Der"

            # --- FILTRO: ambos puntos ≥ 0.75 ---
            if person_conf[selected[0]] < 0.75 or person_conf[selected[1]] < 0.75:
                continue

            # Coordenadas y confianzas (en el frame reducido)
            x_h, y_h = person_kp[selected[0]]
            conf_h = person_conf[selected[0]]
            x_c, y_c = person_kp[selected[1]]
            conf_c = person_conf[selected[1]]

            personas_detectadas.append((pid, lado, x_h, y_h, conf_h, x_c, y_c, conf_c))

    # ----------- FIN cronómetro -----------
    t1 = time.time()
    tiempo_ms = (t1 - t0) * 1000

    # Guardar en CSV
    for pid, lado, x_h, y_h, conf_h, x_c, y_c, conf_c in personas_detectadas:
        csv_writer.writerow([
            frame_id, f"{tiempo_ms:.2f}", pid, lado,
            f"{x_h:.2f}", f"{y_h:.2f}", f"{conf_h:.2f}",
            f"{x_c:.2f}", f"{y_c:.2f}", f"{conf_c:.2f}"
        ])

    print(f"Frame {frame_id} procesado en {tiempo_ms:.2f} ms, personas válidas: {len(personas_detectadas)}")

    # ----------- DIBUJO (fuera del tiempo) -----------
    for pid, lado, x_h, y_h, conf_h, x_c, y_c, conf_c in personas_detectadas:
        cv2.circle(frame_resized, (int(x_h), int(y_h)), 6, (0, 255, 0), -1)
        cv2.circle(frame_resized, (int(x_c), int(y_c)), 6, (0, 255, 0), -1)
        cv2.putText(frame_resized, f"{lado} {pid} conf_h={conf_h:.2f} conf_c={conf_c:.2f}", 
                    (int(x_h), int(y_h) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Keypoints seleccionados (resized antes de YOLO)", frame_resized)
    cv2.waitKey(0)  # avanzar manualmente

    frame_id += 1

cap.release()
csv_file.close()
cv2.destroyAllWindows()
