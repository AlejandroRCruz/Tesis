import cv2

# === 1. Leer la imagen ===
img = cv2.imread("imagen_mensaje.png")
if img is None:
    print("No se pudo abrir la imagen")
    exit()

# === 2. Convertir a RGB (OpenCV usa BGR por defecto) ===
print(img)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
alto, ancho, _ = rgb.shape
print(f"Imagen cargada: {ancho}x{alto} píxeles")

# === 3. Guardar en formato matriz ===
archivo = "imagen_rgb.txt"

with open(archivo, "w") as f:
    for y in range(alto):
        fila = ""
        for x in range(ancho):
            r, g, b = rgb[y, x]
            fila += f"({r},{g},{b}) "
        f.write(fila.strip() + "\n")

print(f"Archivo '{archivo}' creado correctamente en formato de matriz RGB.")
