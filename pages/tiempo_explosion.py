import streamlit as st
import numpy as np
import altair as alt

st.set_page_config(page_title="Escape del Proceso Estocástico", layout="wide")
st.title("⏱️ Movimiento Browniano y Tiempos de Escape")

# --- Parámetros ---
st.sidebar.header("⚙️ Parámetros")
n_steps = st.sidebar.slider("Número de pasos", 100, 2000, 500, step=100)
dt = st.sidebar.slider("Δt", 0.001, 0.1, 0.01)
n_particles = st.sidebar.slider("Número de partículas", 1, 5, 2)
max_n = st.sidebar.slider("Máximo n para los intervalos [1/n, n]", 1, 5, 3)
seed = st.sidebar.number_input("Semilla aleatoria", value=0, min_value=0)
np.random.seed(seed if seed != 0 else None)

# --- Simulación ---
X = np.zeros((n_steps, n_particles))
for i in range(n_particles):
    dW = np.sqrt(dt) * np.random.randn(n_steps)
    X[:, i] = np.cumsum(dW)

time_points = np.arange(n_steps) * dt

# --- Preparar datos para Altair ---
chart_data = []
for i in range(n_particles):
    for t, x in zip(time_points, X[:, i]):
        chart_data.append({"Tiempo": t, "Posición": x, "Partícula": f"Partícula {i+1}", "Escape": False})

# --- Detectar escapes y marcar ---
for n in range(1, max_n + 1):
    lower, upper = 1/n, n
    for i in range(n_particles):
        for idx, t in enumerate(time_points):
            if X[idx, i] < lower or X[idx, i] > upper:
                chart_data.append({
                    "Tiempo": t,
                    "Posición": X[idx, i],
                    "Partícula": f"Partícula {i+1}",
                    "Escape": True
                })

# --- Convertir a NumPy estructurado para Altair ---
import pandas as pd
df = pd.DataFrame(chart_data)

# --- Graficar ---
base = alt.Chart(df).mark_line().encode(
    x='Tiempo',
    y='Posición',
    color='Partícula'
)

# Puntos de escape
points = alt.Chart(df[df['Escape']]).mark_point(shape='cross', size=60, color='red').encode(
    x='Tiempo',
    y='Posición',
    tooltip=['Partícula', 'Tiempo', 'Posición']
)

st.altair_chart(base + points, use_container_width=True)
