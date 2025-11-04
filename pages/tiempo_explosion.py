import streamlit as st
import numpy as np
import altair as alt

st.title("Movimiento Browniano y Tiempos de Escape")

# Parámetros
n_steps = st.slider("Número de pasos", 100, 2000, 500)
dt = st.slider("Δt", 0.001, 0.1, 0.01)
n_particles = st.slider("Número de partículas", 1, 3, 2)
max_n = st.slider("Máximo n para intervalos [1/n, n]", 1, 5, 3)
seed = st.number_input("Semilla aleatoria", value=0)
np.random.seed(seed if seed != 0 else None)

# Simulación del movimiento browniano
X = np.zeros((n_steps, n_particles))
for i in range(n_particles):
    X[:, i] = np.cumsum(np.sqrt(dt) * np.random.randn(n_steps))
time_points = np.arange(n_steps) * dt

# Preparar datos de trayectoria
chart_data = []
for i in range(n_particles):
    for t, x in zip(time_points, X[:, i]):
        chart_data.append({"Tiempo": t, "Posición": x, "Partícula": f"Partícula {i+1}", "Escape": False})

# Detectar solo los puntos donde escapa de **algún** intervalo [1/n, n]
escape_points = []
for n in range(1, max_n + 1):
    lower, upper = 1/n, n
    for i in range(n_particles):
        for idx, x_val in enumerate(X[:, i]):
            if x_val < lower or x_val > upper:
                escape_points.append({
                    "Tiempo": time_points[idx],
                    "Posición": x_val,
                    "Partícula": f"Partícula {i+1}"
                })
escape_df = alt.Data(values=escape_points)

# Graficar
base = alt.Chart(chart_data).mark_line().encode(
    x='Tiempo',
    y='Posición',
    color='Partícula'
)
points = alt.Chart(escape_df).mark_point(shape='cross', color='red', size=60).encode(
    x='Tiempo',
    y='Posición',
    tooltip=['Partícula', 'Tiempo', 'Posición']
)
st.altair_chart(base + points, use_container_width=True)

