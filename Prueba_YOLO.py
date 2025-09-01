import cv2
from ultralytics import YOLO

# Cargar modelo preentrenado YOLOv8 pose
model = YOLO("yolov8s-pose.pt")  # Modelo pequeño y rápido

# Abrir cámara web (puedes cambiarlo a una ruta de video o imagen)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar keypoints en el frame
    results = model(frame, verbose=False)

    # Procesar resultados
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()  # Coordenadas (x, y)
        if len(keypoints) > 0:
            for person in keypoints:
                # Índices de keypoints en COCO (YOLOv8 usa este orden):
                # 5 -> hombro izquierdo, 6 -> hombro derecho
                # 11 -> cadera izquierda, 12 -> cadera derecha
                left_shoulder = tuple(person[5].astype(int))
                right_shoulder = tuple(person[6].astype(int))
                left_hip = tuple(person[11].astype(int))
                right_hip = tuple(person[12].astype(int))

                # Dibujar puntos
                cv2.circle(frame, left_shoulder, 5, (0, 255, 0), -1)
                cv2.circle(frame, right_shoulder, 5, (0, 255, 0), -1)
                cv2.circle(frame, left_hip, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_hip, 5, (255, 0, 0), -1)

                # Dibujar líneas para mayor claridad
                cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 255), 2)
                cv2.line(frame, left_hip, right_hip, (255, 255, 0), 2)

    # Mostrar frame procesado
    cv2.imshow("Detección de Hombros y Cadera", frame)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
