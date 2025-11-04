import streamlit as st
import numpy as np

# Configuración inicial
st.set_page_config(page_title="Movimiento Browniano 1D", layout="wide")
st.title("🎲 Simulación del Movimiento Browniano 1D")

# Sidebar con parámetros
st.sidebar.header("⚙️ Parámetros de simulación")

n_steps = st.sidebar.slider("Número de pasos", 100, 2000, 500, step=100)
dt = st.sidebar.slider("Tamaño del paso (Δt)", 0.001, 1.0, 0.01)
n_particles = st.sidebar.slider("Número de partículas", 1, 10, 3)
show_mean = st.sidebar.checkbox("Mostrar trayectoria promedio", True)
animate = st.sidebar.checkbox("Animar paso a paso", False)

# Semilla
seed = st.sidebar.number_input("Semilla aleatoria (opcional)", value=0, min_value=0, step=1)
np.random.seed(seed if seed != 0 else None)

# Función del movimiento browniano
def brownian_motion_1d(n_steps, dt, n_particles):
    dW = np.sqrt(dt) * np.random.randn(n_steps, n_particles)
    X = np.cumsum(dW, axis=0)
    return X

# Simulación
X = brownian_motion_1d(n_steps, dt, n_particles)
time_points = np.arange(n_steps) * dt

# Preparar gráfico
st.subheader("📈 Trayectorias del Movimiento Browniano")

# Animación o gráfico completo
if animate:
    chart = st.line_chart()
    for i in range(1, n_steps + 1):
        data = {f"Partícula {j+1}": X[:i, j] for j in range(n_particles)}
        if show_mean:
            data["Promedio"] = np.mean(X[:i, :], axis=1)
        data["Tiempo"] = time_points[:i]
        chart.add_rows(data)
else:
    data = {f"Partícula {j+1}": X[:, j] for j in range(n_particles)}
    if show_mean:
        data["Promedio"] = np.mean(X, axis=1)
    data["Tiempo"] = time_points
    st.line_chart(data)

# Explicación
st.markdown("""
---
### 📘 Explicación
El **movimiento browniano unidimensional** describe cómo una partícula se desplaza de forma aleatoria, 
donde cada incremento sigue una distribución normal:

$$ X_{t+Δt} = X_t + N(0, Δt) $$

- Cada línea representa una partícula diferente.  
- La línea discontinua (si está activada) muestra el promedio de todas.  
- Puedes activar la opción de animación para ver cómo evolucionan las trayectorias paso a paso.
""")
