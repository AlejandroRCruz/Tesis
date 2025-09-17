import cv2
import csv

# ---------- Config ----------
VIDEO_PATH = r"C:/Users/Alejandro/Documents/GitHub/Tesis/Videos Necesarios/Prueba_Eri_distancia.mp4"
CSV_PATH = "segmentos_frames.csv"

# ---------- Setup ----------
cap = cv2.VideoCapture(VIDEO_PATH)

segmentos = []   # {"name": str, "start": int, "end": int}
segment_start = None

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# CSV (sobrescribe encabezado)
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["name", "start_frame", "end_frame"])

frame_idx = 1  # 👈 empieza en Frame 1 (humano)

def mostrar_frame(idx):
    """Muestra un frame específico (1-based para el usuario)."""
    if idx < 1 or idx > total_frames:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx-1)  # OpenCV usa base 0
    ok, frame = cap.read()
    if not ok:
        return None
    # overlay
    cv2.putText(frame, f"Frame: {idx}/{total_frames}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if segment_start is not None:
        cv2.putText(frame, f"Marcando desde {segment_start}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "Teclas: [s]=inicio [e]=fin [u]=undo [<-]=atras [->]=adelante [q]=salir",
                (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    return frame

# Códigos de flechas (Windows con waitKeyEx)
KEY_LEFTS  = {2424832, 81}
KEY_RIGHTS = {2555904, 83}

while True:
    frame = mostrar_frame(frame_idx)
    if frame is None:
        print("Fin del video o no se pudo leer el frame.")
        break

    cv2.imshow("Segmentador de Frames", frame)

    key = cv2.waitKeyEx(0)

    if key == ord('q'):
        print("Saliendo...")
        break

    elif key == ord('s'):
        segment_start = frame_idx
        print(f"[INICIO] Segmento empieza en frame {segment_start}")

    elif key == ord('e'):
        if segment_start is None:
            print("⚠️ Marca un inicio con 's' antes de finalizar.")
        else:
            segment_end = frame_idx
            if segment_end < segment_start:
                segment_start, segment_end = segment_end, segment_start
            name = input(f"Nombre del segmento ({segment_start}-{segment_end}): ").strip() or f"segmento_{segment_start}_{segment_end}"
            segmentos.append({"name": name, "start": segment_start, "end": segment_end})
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([name, segment_start, segment_end])
            print(f"[GUARDADO] '{name}': {segment_start} -> {segment_end}")
            segment_start = None

    elif key == ord('u'):
        if not segmentos:
            print("No hay segmentos para deshacer.")
        else:
            ultimo = segmentos.pop()
            print(f"[UNDO] Eliminado: {ultimo}")
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["name", "start_frame", "end_frame"])
                for s in segmentos:
                    w.writerow([s["name"], s["start"], s["end"]])

    elif key in KEY_LEFTS:
        frame_idx = max(1, frame_idx - 1)  # 👈 ahora nunca baja de 1

    elif key in KEY_RIGHTS:
        frame_idx = min(total_frames, frame_idx + 1)

    else:
        frame_idx = min(total_frames, frame_idx + 1)

cap.release()
cv2.destroyAllWindows()

print("\n✅ Segmentos registrados:")
for s in segmentos:
    print(f" - {s['name']}: {s['start']} -> {s['end']}")
print(f"\n📁 CSV guardado en: {CSV_PATH}")
