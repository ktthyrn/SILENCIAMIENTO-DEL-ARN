import streamlit as st
import numpy as np

st.set_page_config(page_title="Tiempos de Escape del Proceso Estocástico", layout="wide")
st.title("⏱️ Tiempos de Escape del Movimiento Browniano")

# --- Parámetros ---
st.sidebar.header("⚙️ Parámetros de simulación")
n_steps = st.sidebar.slider("Número de pasos", 100, 5000, 1000, step=100)
dt = st.sidebar.slider("Δt", 0.001, 0.1, 0.01)
n_particles = st.sidebar.slider("Número de partículas", 1, 5, 2)
max_n = st.sidebar.slider("Máximo n para los intervalos [1/n, n]", 1, 20, 5)
seed = st.sidebar.number_input("Semilla aleatoria", value=0, min_value=0)
np.random.seed(seed if seed != 0 else None)

# --- Simulación del movimiento browniano ---
X = np.zeros((n_steps, n_particles))
for i in range(n_particles):
    dW = np.sqrt(dt) * np.random.randn(n_steps)
    X[:, i] = np.cumsum(dW)

time_points = np.arange(n_steps) * dt

# --- Detección de tiempos de escape ---
st.subheader("⏳ Tiempos de escape del intervalo [1/n, n]")

escape_dict = {}
for n in range(1, max_n + 1):
    lower, upper = 1/n, n
    escape_times = []
    for i in range(n_particles):
        escapes = time_points[(X[:, i] < lower) | (X[:, i] > upper)]
        escape_times.append(escapes)
    escape_dict[n] = escape_times

# --- Mostrar resultados ---
for n in range(1, max_n + 1):
    st.markdown(f"**n = {n}, intervalo = [{1/n:.3f}, {n}]**")
    for i, times in enumerate(escape_dict[n]):
        if len(times) > 0:
            st.write(f"Partícula {i+1}: tiempos de escape ≈ {times}")
        else:
            st.write(f"Partícula {i+1}: no salió del intervalo")
    st.write("---")

# --- Visualización de las trayectorias ---
st.subheader("📈 Trayectorias del movimiento browniano")
columns = [f"Partícula {i+1}" for i in range(n_particles)]
data = {columns[i]: X[:, i] for i in range(n_particles)}
data["Tiempo"] = time_points
st.line_chart(data)
