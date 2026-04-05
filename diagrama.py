import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ------------------------------
# Definición de tareas y fechas
# ------------------------------
tasks = [
    "Revisión técnica y exploratoria",
    "Propuesta de arquitectura preliminar",
    "Desarrollo del módulo de percepción visual",
    "Cálculo de ubicación espacial",
    "Síntesis de audio espacial informativo",
    "Prueba en entorno real, del dispositivo",
    "Mejoras y ajustes del dispositivo",
    "Evaluación iterativa con usuarios reales",
    "Escritura de Tesis",
    "Revisión de Tesis",
]

start_dates = [
    datetime(2025, 2, 1),   # Revisión técnica
    datetime(2025, 3, 1),   # Propuesta arquitectura
    datetime(2025, 5, 1),   # Desarrollo percepción
    datetime(2025, 7, 1),   # Cálculo ubicación

    datetime(2025, 12, 1),  # Síntesis audio (ajustado)
    datetime(2025, 12, 1),  # Prueba en entorno real (ajustado)
    datetime(2025, 11, 1),  # Mejoras dispositivo (ajustado)

    datetime(2026, 1, 1),   # Evaluación usuarios
    datetime(2025, 11, 1),  # Escritura tesis (ajustado)
    datetime(2026, 1, 1),   # Revisión tesis
]

end_dates = [
    datetime(2025, 6, 30),  # Revisión técnica
    datetime(2025, 6, 30),  # Propuesta arquitectura
    datetime(2025, 10, 31), # Desarrollo percepción
    datetime(2025, 10, 31), # Cálculo ubicación

    datetime(2026, 2, 28),  # Síntesis audio (dic 25 – feb 26)
    datetime(2026, 3, 31),  # Prueba entorno real (dic 25 – mar 26)
    datetime(2026, 4, 30),  # Mejoras dispositivo (nov 25 – abr 26)

    datetime(2026, 6, 30),  # Evaluación usuarios
    datetime(2026, 5, 31),  # Escritura tesis (nov 25 – may 26)
    datetime(2026, 5, 30),  # Revisión tesis
]

# Duración en días
durations = [(e - s).days for s, e in zip(start_dates, end_dates)]

# ------------------------------
# Gráfica tipo Gantt
# ------------------------------
fig, ax = plt.subplots(figsize=(12, 5))

y_pos = range(len(tasks))

ax.barh(
    y_pos,
    durations,
    left=start_dates,
    align="center"
)

ax.set_yticks(y_pos)
ax.set_yticklabels(tasks)
ax.invert_yaxis()

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

plt.xticks(rotation=45, ha="right")
ax.set_title("Cronograma del Proyecto (Feb 2025 - Jun 2026)")
ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
