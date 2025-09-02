import cv2
from ultralytics import YOLO

# Cargar modelo YOLOv8 Pose
model = YOLO("yolov8s-pose.pt")  # Puedes usar otro modelo si quieres

# Cargar el video (cambia "mi_video.mp4" por el nombre real)
path= 'C:/Users/Alejandro/Documents/GitHub/Tesis/Videos Necesarios/Prueba_Eri_distancia.mp4'
cap = cv2.VideoCapture(path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detección de keypoints
    results = model(frame, verbose=False)

    # Procesar resultados
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        if len(keypoints) > 0:
            for person in keypoints:
                # Índices COCO: 5 hombro izq, 6 hombro der, 11 cadera izq, 12 cadera der
                left_shoulder = tuple(person[5].astype(int))
                right_shoulder = tuple(person[6].astype(int))
                left_hip = tuple(person[11].astype(int))
                right_hip = tuple(person[12].astype(int))

                # Dibujar puntos
                cv2.circle(frame, left_shoulder, 5, (0, 255, 0), -1)
                cv2.circle(frame, right_shoulder, 5, (0, 255, 0), -1)
                cv2.circle(frame, left_hip, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_hip, 5, (255, 0, 0), -1)

                # Opcional: dibujar líneas para mayor claridad
                cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 255), 2)
                cv2.line(frame, left_hip, right_hip, (255, 255, 0), 2)

    # Mostrar video en ventana
    cv2.imshow("Detección de Hombros y Cadera", frame)

    # Presiona "q" para salir manualmente antes de que acabe
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
