import cv2
import csv
from ultralytics import YOLO

# ----- Config -----
MODEL_PATH = "yolov8n-pose.pt"
VIDEO_PATH = r"C:/Users/Alejandro/Documents/GitHub/Tesis/Videos Necesarios/Prueba_Eri_distancia.mp4"
CSV_PATH = "coordenadas_hombros_caderas.csv"

# Índices COCO
IDX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12
}

# ----- Cargar modelo y video -----
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# ----- CSV -----
with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame", "person_id",
        "left_shoulder_x", "left_shoulder_y", "left_shoulder_conf",
        "right_shoulder_x", "right_shoulder_y", "right_shoulder_conf",
        "left_hip_x", "left_hip_y", "left_hip_conf",
        "right_hip_x", "right_hip_y", "right_hip_conf"
    ])

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Inferencia
        results = model(frame, verbose=False)

        for res in results:
            if res.keypoints is None:
                continue

            xy = res.keypoints.xy.cpu().numpy()
            conf = res.keypoints.conf.cpu().numpy()

            num_personas = xy.shape[0]

            for pid in range(num_personas):
                # Extraer puntos
                ls = (int(xy[pid, IDX["left_shoulder"], 0]), int(xy[pid, IDX["left_shoulder"], 1]))
                rs = (int(xy[pid, IDX["right_shoulder"], 0]), int(xy[pid, IDX["right_shoulder"], 1]))
                lh = (int(xy[pid, IDX["left_hip"], 0]), int(xy[pid, IDX["left_hip"], 1]))
                rh = (int(xy[pid, IDX["right_hip"], 0]), int(xy[pid, IDX["right_hip"], 1]))

                # Confiabilidad
                ls_c = float(conf[pid, IDX["left_shoulder"]])
                rs_c = float(conf[pid, IDX["right_shoulder"]])
                lh_c = float(conf[pid, IDX["left_hip"]])
                rh_c = float(conf[pid, IDX["right_hip"]])

                # Dibujar
                cv2.circle(frame, ls, 5, (0, 255, 0), -1)
                cv2.circle(frame, rs, 5, (0, 255, 0), -1)
                cv2.circle(frame, lh, 5, (255, 0, 0), -1)
                cv2.circle(frame, rh, 5, (255, 0, 0), -1)
                cv2.line(frame, ls, rs, (0, 255, 255), 2)
                cv2.line(frame, lh, rh, (255, 255, 0), 2)

                # Guardar fila en CSV
                writer.writerow([
                    frame_idx, pid,
                    ls[0], ls[1], ls_c,
                    rs[0], rs[1], rs_c,
                    lh[0], lh[1], lh_c,
                    rh[0], rh[1], rh_c
                ])

        # Mostrar frame en tiempo real
        cv2.imshow("Hombros y Caderas", frame)

        # Avanzar automáticamente (25 ms entre frames, aprox 40 fps)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ CSV guardado en: {CSV_PATH}")
