import streamlit as st
import numpy as np

st.set_page_config(page_title="Movimiento Browniano 1D", layout="wide")
st.title("🎲 Movimiento Browniano 1D (versión estable)")

# --- Parámetros ---
st.sidebar.header("⚙️ Parámetros de simulación")
n_steps = st.sidebar.slider("Número de pasos", 100, 2000, 500, step=100)
dt = st.sidebar.slider("Tamaño del paso (Δt)", 0.001, 1.0, 0.01)
n_particles = st.sidebar.slider("Número de partículas", 1, 10, 3)
show_mean = st.sidebar.checkbox("Mostrar trayectoria promedio", True)
animate = st.sidebar.checkbox("Animar paso a paso", False)
seed = st.sidebar.number_input("Semilla aleatoria (opcional)", value=0, min_value=0, step=1)

# --- Simulación ---
np.random.seed(seed if seed != 0 else None)
def brownian_motion_1d(n_steps, dt, n_particles):
    dW = np.sqrt(dt) * np.random.randn(n_steps, n_particles)
    return np.cumsum(dW, axis=0)

X = brownian_motion_1d(n_steps, dt, n_particles)
time_points = np.arange(n_steps) * dt

# --- Preparar columnas fijas ---
columns = [f"Partícula {i+1}" for i in range(n_particles)]
if show_mean:
    columns.append("Promedio")

# --- Crear placeholder y gráfico ---
st.subheader("📈 Trayectorias del Movimiento Browniano")
chart_placeholder = st.empty()

if animate:
    # Initialize empty array with fixed columns (avoids axis jump)
    data = {col: [] for col in columns}
    chart = chart_placeholder.line_chart(data)

    for i in range(1, n_steps + 1):
        frame = np.column_stack([
            X[:i, j] for j in range(n_particles)
        ])
        if show_mean:
            mean_col = np.mean(frame, axis=1)[:, None]
            frame = np.hstack([frame, mean_col])

        # Use dict mapping for stable update
        new_data = {columns[k]: frame[:, k] for k in range(len(columns))}
        chart.add_rows(new_data)
else:
    frame = np.column_stack([
        X[:, j] for j in range(n_particles)
    ])
    if show_mean:
        mean_col = np.mean(frame, axis=1)[:, None]
        frame = np.hstack([frame, mean_col])
    data = {columns[k]: frame[:, k] for k in range(len(columns))}
    chart_placeholder.line_chart(data)

# --- Explicación ---
st.markdown("""
---
### 📘 Explicación
El **movimiento browniano unidimensional** describe un proceso aleatorio donde cada incremento sigue una distribución normal:

$$ X_{t+Δt} = X_t + N(0, Δt) $$

Esta versión mantiene el gráfico **estable** durante la animación,
evitando que los ejes cambien o el gráfico se desplace.
""")
