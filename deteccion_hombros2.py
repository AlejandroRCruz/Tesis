import cv2
import csv
from ultralytics import YOLO

# Cargar modelo YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Cargar video
path = 'C:/Users/Alejandro/Documents/GitHub/Tesis/Videos Necesarios/Prueba_Eri_distancia.mp4'
cap = cv2.VideoCapture(path)

# Crear archivo CSV
csv_file = "coordenadas_keypoints.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Encabezados
    writer.writerow([
        "frame",
        "left_shoulder_x", "left_shoulder_y",
        "right_shoulder_x", "right_shoulder_y",
        "left_hip_x", "left_hip_y",
        "right_hip_x", "right_hip_y"
    ])

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # contador de frames

        # Detección de keypoints
        results = model(frame, verbose=False)

        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()

            for person in keypoints:
                # Extraer keypoints con enteros puros
                left_shoulder = (int(person[5][0]), int(person[5][1]))
                right_shoulder = (int(person[6][0]), int(person[6][1]))
                left_hip = (int(person[11][0]), int(person[11][1]))
                right_hip = (int(person[12][0]), int(person[12][1]))

                # Dibujar en el frame
                cv2.circle(frame, left_shoulder, 5, (0, 255, 0), -1)
                cv2.circle(frame, right_shoulder, 5, (0, 255, 0), -1)
                cv2.circle(frame, left_hip, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_hip, 5, (255, 0, 0), -1)
                cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 255), 2)
                cv2.line(frame, left_hip, right_hip, (255, 255, 0), 2)

                # Imprimir en consola
                print(f"Frame {frame_count} -> "
                      f"Hombro Izq: {left_shoulder}, Hombro Der: {right_shoulder}, "
                      f"Cadera Izq: {left_hip}, Cadera Der: {right_hip}")

                # Guardar en CSV
                writer.writerow([
                    frame_count,
                    left_shoulder[0], left_shoulder[1],
                    right_shoulder[0], right_shoulder[1],
                    left_hip[0], left_hip[1],
                    right_hip[0], right_hip[1]
                ])

        # Mostrar frame
        cv2.imshow("Hombros y Caderas", frame)

        # Avanzar manualmente
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Coordenadas guardadas en: {csv_file}")
