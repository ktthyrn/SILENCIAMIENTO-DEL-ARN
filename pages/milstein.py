import streamlit as st
import numpy as np
import pandas as pd

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide")

st.title("🔬 Comparación: Determinista vs. Milstein")
st.write("""
Utiliza los parámetros de la barra lateral para ajustar el modelo.
Cada gráfico compara el modelo determinista (línea oscura) con 
**una sola realización** estocástica del Método de Milstein (línea clara).
""")

# --- 2. BARRA LATERAL (CON SLIDERS Y LLAVES ÚNICAS) ---
st.sidebar.header("⚙️ Parámetros de Simulación (RNAi)")

# NOTA: Cada 'key' es única (ej: "mil_n") para evitar errores
with st.sidebar.expander("Parámetros del Modelo (ι)", expanded=True):
    col1, col2 = st.columns(2)
    n = col1.slider("n (tasa RISC)", 1, 20, 5, key="mil_n")
    iota_a = col1.slider("ι_a", 0.0, 20.0, 10.0, format="%.1f", key="mil_ia")
    iota_b = col1.slider("ι_b", 0.0, 0.01, 0.001, format="%.4f", key="mil_ib")
    iota_c = col1.slider("ι_c", 0.0, 5.0, 1.0, format="%.1f", key="mil_ic")
    iota_h = col2.slider("ι_h", 500.0, 2000.0, 1000.0, key="mil_ih")
    iota_g = col2.slider("ι_g", 0.0, 5.0, 1.0, format="%.1f", key="mil_ig")
    iota_m = col2.slider("ι_m", 0.0, 5.0, 1.0, format="%.1f", key="mil_im")
    iota_r = col2.slider("ι_r", 0.0, 1.0, 0.1, format="%.2f", key="mil_ir")

with st.sidebar.expander("Condiciones Iniciales", expanded=True):
    col1, col2 = st.columns(2)
    S0 = col1.slider("S_0 (dsRNA)", 0.0, 100.0, 10.0, key="mil_s0")
    R0 = col1.slider("R_0 (RISC)", 0.0, 100.0, 0.0, key="mil_r0")
    C0 = col2.slider("C_0 (Complejo)", 0.0, 100.0, 0.0, key="mil_c0")
    M0 = col2.slider("M_0 (mRNA)", 500.0, 2000.0, 1000.0, key="mil_m0")

with st.sidebar.expander("Parámetros de Simulación", expanded=True):
    # ¡VALOR POR DEFECTO 0.01 (PRECISO)!
    T = st.slider("Tiempo Total (T)", 10.0, 200.0, 50.0, key="mil_T")
    dt = st.slider("Paso (dt)", 0.001, 0.1, 0.01, format="%.3f", key="mil_dt") # Default 0.01

with st.sidebar.expander("Intensidad de Ruido (σ)", expanded=True):
    col1, col2 = st.columns(2)
    sigma1 = col1.slider("σ_1", 0.0, 1.0, 0.1, format="%.2f", key="mil_s1")
    sigma2 = col1.slider("σ_2", 0.0, 1.0, 0.1, format="%.2f", key="mil_s2")
    sigma3 = col2.slider("σ_3", 0.0, 1.0, 0.1, format="%.2f", key="mil_s3")
    sigma4 = col1.slider("σ_4", 0.0, 1.0, 0.1, format="%.2f", key="mil_s4")


# --- 3. FUNCIÓN DE SIMULACIÓN (CON LÓGICA DE MILSTEIN) ---
@st.cache_data
def simular_modelos_milstein(n, iota_a, iota_b, iota_c, iota_h, iota_g, iota_m, iota_r,
                             sigma1, sigma2, sigma3, sigma4,
                             S0, R0, C0, M0,
                             T, dt):
    
    # --- Preparación ---
    n_realizations = 1
    N = int(T / dt) + 1 
    t = np.linspace(0, T, N)
    sqrt_dt = np.sqrt(dt)

    # --- Arrays para Euler Determinista (1D) ---
    S_det, R_det, C_det, M_det = (np.zeros(N) for _ in range(4))
    S_det[0], R_det[0], C_det[0], M_det[0] = S0, R0, C0, M0

    # --- Arrays para Milstein (1D) ---
    S_mil, R_mil, C_mil, M_mil = (np.zeros(N) for _ in range(4))
    S_mil[0], R_mil[0], C_mil[0], M_mil[0] = S0, R0, C0, M0

    # --- Bucle de Simulación ---
    for i in range(1, N):
        
        # --- Modelo Determinista (Euler) ---
        S_prev_d, R_prev_d, C_prev_d, M_prev_d = S_det[i-1], R_det[i-1], C_det[i-1], M_det[i-1]
        
        dS_det_val = (-iota_a * S_prev_d + iota_g * C_prev_d) * dt
        dR_det_val = (n * iota_a * S_prev_d - iota_r * R_prev_d - iota_b * R_prev_d * M_prev_d) * dt
        dC_det_val = (iota_b * R_prev_d * M_prev_d - (iota_g + iota_c) * C_prev_d) * dt
        dM_det_val = (iota_h - iota_m * M_prev_d - iota_b * R_prev_d * M_prev_d) * dt
        
        S_det[i] = max(S_prev_d + dS_det_val, 0)
        R_det[i] = max(R_prev_d + dR_det_val, 0)
        C_det[i] = max(C_prev_d + dC_det_val, 0)
        M_det[i] = max(M_prev_d + dM_det_val, 0)

        # --- Modelo Estocástico (Milstein) ---
        # (Lógica exacta de tu script original, que usa 1 solo ruido 'tau')
        tau = np.random.normal(0, 1) # tau = dW / sqrt(dt)
        S_prev_m, R_prev_m, C_prev_m, M_prev_m = S_mil[i-1], R_mil[i-1], C_mil[i-1], M_mil[i-1]

        # Términos deterministas (drift)
        dS_drift = (-iota_a * S_prev_m + iota_g * C_prev_m) * dt
        dR_drift = (n * iota_a * S_prev_m - iota_r * R_prev_m - iota_b * R_prev_m * M_prev_m) * dt
        dC_drift = (iota_b * R_prev_m * M_prev_m - (iota_g + iota_c) * C_prev_m) * dt
        dM_drift = (iota_h - iota_m * M_prev_m - iota_b * R_prev_m * M_prev_m) * dt

        # Términos de Milstein (copiados de tu script)
        # (tau**2 - 1) * dt  es lo mismo que (dW**2 - dt)
        dS_milstein = (-sigma1 * S_prev_m * sqrt_dt * tau -
                       0.5 * sigma1**2 * S_prev_m * (tau**2 - 1) * dt)

        dR_milstein = (sigma1 * n * S_prev_m * sqrt_dt * tau +
                       0.5 * sigma1**2 * n**2 * S_prev_m * (tau**2 - 1) * dt -
                       sigma2 * R_prev_m * sqrt_dt * tau -
                       0.5 * sigma2**2 * R_prev_m * (tau**2 - 1) * dt)

        dC_milstein = (-sigma3 * C_prev_m * sqrt_dt * tau -
                       0.5 * sigma3**2 * C_prev_m * (tau**2 - 1) * dt)

        dM_milstein = (-sigma4 * M_prev_m * sqrt_dt * tau -
                       0.5 * sigma4**2 * M_prev_m * (tau**2 - 1) * dt)

        # Suma final
        S_mil[i] = max(S_prev_m + dS_drift + dS_milstein, 0)
        R_mil[i] = max(R_prev_m + dR_drift + dR_milstein, 0)
        C_mil[i] = max(C_prev_m + dC_drift + dC_milstein, 0)
        M_mil[i] = max(M_prev_m + dM_drift + dM_milstein, 0)

    return S_det, R_det, C_det, M_det, S_mil, R_mil, C_mil, M_mil, t

# --- 4. FUNCIÓN AUXILIAR PARA "DIEZMAR" ---
def downsample_dataframe(df, max_points=1000):
    """Reduce el número de puntos en un dataframe para graficar rápido."""
    if len(df) > max_points:
        step = len(df) // max_points
        return df.iloc[::step]
    return df

# --- 5. EJECUCIÓN Y GRÁFICOS (SEPARADOS) ---
with st.spinner("Ejecutando simulación precisa (puede tardar un poco)..."):
    
    # 1. Llamamos a la simulación (en caché)
    S_det, R_det, C_det, M_det, S_mil, R_mil, C_mil, M_mil, t = simular_modelos_milstein(
        n=n, iota_a=iota_a, iota_b=iota_b, iota_c=iota_c, iota_h=iota_h,
        iota_g=iota_g, iota_m=iota_m, iota_r=iota_r,
        sigma1=sigma1, sigma2=sigma2, sigma3=sigma3, sigma4=sigma4,
        S0=S0, R0=R0, C0=C0, M0=M0,
        T=T, dt=dt
    )
    
    # 2. Preparamos 4 DataFrames SEPARADOS
    df_S = pd.DataFrame({"S (Determinista)": S_det, "S (Milstein)": S_mil}, index=t)
    df_R = pd.DataFrame({"R (Determinista)": R_det, "R (Milstein)": R_mil}, index=t)
    df_C = pd.DataFrame({"C (Determinista)": C_det, "C (Milstein)": C_mil}, index=t)
    df_M = pd.DataFrame({"M (Determinista)": M_det, "M (Milstein)": M_mil}, index=t)

    # 3. "Diezmamos" los 4 DataFrames para que el gráfico sea rápido
    df_S_plot = downsample_dataframe(df_S)
    df_R_plot = downsample_dataframe(df_R)
    df_C_plot = downsample_dataframe(df_C)
    df_M_plot = downsample_dataframe(df_M)
    
    # 4. Creamos las 2 columnas para los gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 dsRNA (S)")
        st.line_chart(df_S_plot)
        
        st.subheader("📈 Complejo (C)")
        st.line_chart(df_C_plot)
    
    with col2:
        st.subheader("📈 RISC (R)")
        st.line_chart(df_R_plot)
        
        st.subheader("📈 mRNA (M)")
        st.line_chart(df_M_plot)

    st.caption(f"Mostrando {len(df_S_plot)} puntos de {len(df_S)} puntos simulados para optimizar el rendimiento.")