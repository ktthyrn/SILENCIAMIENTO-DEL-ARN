import streamlit as st
import numpy as np
import altair as alt
import pandas as pd

st.set_page_config(page_title="Escape del Proceso Estocástico", layout="wide")
st.title("⏱️ Movimiento Browniano y Tiempos de Escape")

# --- Parámetros ---
st.sidebar.header("⚙️ Parámetros")
n_steps = st.sidebar.slider("Número de pasos", 100, 20000, 500, step=100)
dt = st.sidebar.slider("Δt", 0.001, 0.1, 0.01)
# n_particles = st.sidebar.slider("Número de partículas", 1, 5, 2)
n_particles=1
max_n = st.sidebar.slider("Máximo n para los intervalos [1/n, n]", 1, 100, 3)
seed = st.sidebar.number_input("Semilla aleatoria", value=0, min_value=0)
np.random.seed(seed if seed != 0 else None)

# --- Simulación ---
time_points = np.arange(n_steps) * dt
X = np.zeros((n_steps, n_particles))

for i in range(n_particles):
    dW = np.sqrt(dt) * np.random.randn(n_steps)/10
    X[:, i] = 1 + np.cumsum(dW)  # empieza en 1


# --- Preparar datos para Altair -
df_lines = pd.DataFrame({
    "Tiempo": np.tile(time_points, n_particles),
    "Posición": X.flatten(),
    "Partícula": np.repeat([f"Partícula {i+1}" for i in range(n_particles)], n_steps)
})

# --- Detectar escapes ---
escape_points = []
seen = set()  # para evitar duplicados exactos (tiempo, posición, partícula)

import pandas as pd
import numpy as np

max_n=6
n_particles=1
n_steps = 10000
dt=0.01

# --- Simulación ---
time_points = np.arange(n_steps) * dt
X = np.zeros((n_steps, n_particles))

for i in range(n_particles):
    dW = np.sqrt(dt) * np.random.randn(n_steps)/10
    X[:, i] = 1 + np.cumsum(dW)  # empieza en 1

# --- Preparar datos para Altair -
df_lines = pd.DataFrame({
    "Tiempo": np.tile(time_points, n_particles),
    "Posición": X.flatten(),
    "Partícula": np.repeat([f"Partícula {i+1}" for i in range(n_particles)], n_steps)
})
# --- Detectar escapes ---
escape_points = []
seen = set()  # para evitar duplicados exactos (tiempo, posición, partícula)

for n in range(2, max_n + 1):
    lower, upper = 1/n, n
    for i in range(n_particles):
        escape_idx = np.where((X[:, i] < lower) | (X[:, i] > upper))[0]
        if escape_idx.size > 0:
            idx = int(escape_idx[0])
            key = (idx, i)  # indice temporal y partícula
            if key not in seen:
                seen.add(key)
                escape_points.append({
                    "Tiempo": time_points[idx],
                    "Posición": X[idx, i],
                    "Partícula": f"Partícula {i+1}",
                    "n": n
                })

df_escapes = pd.DataFrame(escape_points)


#escape_points
# --- Graficar ---
base = alt.Chart(df_lines).mark_line().encode(
    x='Tiempo',
    y='Posición',
    color='Partícula'
)

# Only add escape points if there are any
if not df_escapes.empty:
    points = alt.Chart(df_escapes).mark_point(shape='cross', size=60, color='red').encode(
        x='Tiempo',
        y='Posición',
        tooltip=['Partícula', 'Tiempo', 'Posición']
    )
    chart = base + points
else:
    chart = base

st.altair_chart(chart, use_container_width=True)
