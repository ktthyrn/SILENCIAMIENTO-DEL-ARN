import streamlit as st
import numpy as np
import pandas as pd  # Para los DataFrames

# --- 1. CONFIGURACIÓN DE PÁGINA (Para gráficos "grandes") ---
st.set_page_config(layout="wide")

st.title("🔬 Comparación: Euler Determinista vs. Milstein")
st.write("""
El Método de Milstein añade un término de corrección al de Euler-Maruyama, 
siendo (generalmente) más preciso.
""")

# --- 2. BARRA LATERAL (¡CON SLIDERS!) ---
# (Estos son los mismos parámetros, así que puedes copiarlos y pegarlos)
st.sidebar.header("⚙️ Parámetros de Simulación (RNAi)")

with st.sidebar.expander("Parámetros del Modelo (ι)", expanded=True):
    col1, col2 = st.columns(2)
    n = col1.slider("n (tasa RISC)", 1, 20, 5, key="milstein_n")
    iota_a = col1.slider("ι_a", 0.0, 20.0, 10.0, format="%.1f", key="milstein_ia")
    iota_b = col1.slider("ι_b", 0.0, 0.01, 0.001, format="%.4f", key="milstein_ib")
    iota_c = col1.slider("ι_c", 0.0, 5.0, 1.0, format="%.1f", key="milstein_ic")
    iota_h = col2.slider("ι_h", 500.0, 2000.0, 1000.0, key="milstein_ih")
    iota_g = col2.slider("ι_g", 0.0, 5.0, 1.0, format="%.1f", key="milstein_ig")
    iota_m = col2.slider("ι_m", 0.0, 5.0, 1.0, format="%.1f", key="milstein_im")
    iota_r = col2.slider("ι_r", 0.0, 1.0, 0.1, format="%.2f", key="milstein_ir")

with st.sidebar.expander("Condiciones Iniciales", expanded=True):
    col1, col2 = st.columns(2)
    S0 = col1.slider("S_0 (dsRNA)", 0.0, 100.0, 10.0, key="milstein_s0")
    R0 = col1.slider("R_0 (RISC)", 0.0, 100.0, 0.0, key="milstein_r0")
    C0 = col2.slider("C_0 (Complejo)", 0.0, 100.0, 0.0, key="milstein_c0")
    M0 = col2.slider("M_0 (mRNA)", 500.0, 2000.0, 1000.0, key="milstein_m0")

with st.sidebar.expander("Parámetros de Simulación", expanded=True):
    col1, col2 = st.columns(2)
    T = col1.slider("Tiempo Total (T)", 10.0, 200.0, 50.0, key="milstein_T")
    dt = col1.slider("Paso (dt)", 0.001, 0.1, 0.01, format="%.3f", key="milstein_dt")
    n_realizations = col2.slider("Nº Realizaciones (Milstein)", 1, 20, 5, key="milstein_n_realiz")

with st.sidebar.expander("Intensidad de Ruido (σ)", expanded=True):
    col1, col2 = st.columns(2)
    sigma1 = col1.slider("σ_1", 0.0, 1.0, 0.1, format="%.2f", key="milstein_s1")
    sigma2 = col1.slider("σ_2", 0.0, 1.0, 0.1, format="%.2f", key="milstein_s2")
    sigma3 = col1.slider("σ_3", 0.0, 1.0, 0.1, format="%.2f", key="milstein_s3")
    sigma4 = col1.slider("σ_4", 0.0, 1.0, 0.1, format="%.2f", key="milstein_s4")


# --- 3. FUNCIÓN DE SIMULACIÓN (CON CACHÉ) ---
@st.cache_data
def simular_modelos_milstein(n, iota_a, iota_b, iota_c, iota_h, iota_g, iota_m, iota_r,
                             sigma1, sigma2, sigma3, sigma4,
                             S0, R0, C0, M0,
                             T, dt, n_realizations):
    
    # --- Preparación ---
    N = int(T / dt) + 1 
    t = np.linspace(0, T, N)
    sqrt_dt = np.sqrt(dt)

    # --- Arrays para Euler Determinista ---
    S_det, R_det, C_det, M_det = (np.zeros(N) for _ in range(4))
    S_det[0], R_det[0], C_det[0], M_det[0] = S0, R0, C0, M0

    # --- Arrays para Milstein ---
    S_mil, R_mil, C_mil, M_mil = (np.zeros((n_realizations, N)) for _ in range(4))
    for r in range(n_realizations):
        S_mil[r, 0], R_mil[r, 0], C_mil[r, 0], M_mil[r, 0] = S0, R0, C0, M0

    # --- Bucle de Simulación ---
    for i in range(1, N):
        
        # --- Modelo Determinista (Euler) ---
        S_prev_d, R_prev_d, C_prev_d, M_prev_d = S_det[i-1], R_det[i-1], C_det[i-1], M_det[i-1]
        
        dS_det = (-iota_a * S_prev_d + iota_g * C_prev_d) * dt
        dR_det = (n * iota_a * S_prev_d - iota_r * R_prev_d - iota_b * R_prev_d * M_prev_d) * dt
        dC_det = (iota_b * R_prev_d * M_prev_d - (iota_g + iota_c) * C_prev_d) * dt
        dM_det = (iota_h - iota_m * M_prev_d - iota_b * R_prev_d * M_prev_d) * dt
        
        S_det[i] = max(S_prev_d + dS_det, 0)
        R_det[i] = max(R_prev_d + dR_det, 0)
        C_det[i] = max(C_prev_d + dC_det, 0)
        M_det[i] = max(M_prev_d + dM_det, 0)

        # --- Modelo Estocástico (Milstein) ---
        for r in range(n_realizations):
            # 4 ruidos independientes
            dB1 = np.random.normal(0, sqrt_dt)
            dB2 = np.random.normal(0, sqrt_dt)
            dB3 = np.random.normal(0, sqrt_dt)
            dB4 = np.random.normal(0, sqrt_dt)
            
            S_prev_m, R_prev_m, C_prev_m, M_prev_m = S_mil[r, i-1], R_mil[r, i-1], C_mil[r, i-1], M_mil[r, i-1]

            # Términos deterministas (drift)
            dS_drift = (-iota_a * S_prev_m + iota_g * C_prev_m) * dt
            dR_drift = (n * iota_a * S_prev_m - iota_r * R_prev_m - iota_b * R_prev_m * M_prev_m) * dt
            dC_drift = (iota_b * R_prev_m * M_prev_m - (iota_g + iota_c) * C_prev_m) * dt
            dM_drift = (iota_h - iota_m * M_prev_m - iota_b * R_prev_m * M_prev_m) * dt

            # Términos estocásticos (diffusión + corrección de Milstein)
            # g(x) = sigma * x  =>  g'(x) = sigma
            # Corrección = 0.5 * g(x) * g'(x) * (dB**2 - dt)
            
            dS_mil = (-sigma1 * S_prev_m * dB1) + \
                     (0.5 * (-sigma1 * S_prev_m) * (-sigma1) * (dB1**2 - dt))
            
            dR_mil = (sigma1 * n * S_prev_m * dB1) + (sigma2 * R_prev_m * dB2) + \
                     (0.5 * (sigma1 * n * S_prev_m) * (sigma1 * n) * (dB1**2 - dt)) + \
                     (0.5 * (sigma2 * R_prev_m) * (sigma2) * (dB2**2 - dt))
            
            dC_mil = (-sigma3 * C_prev_m * dB3) + \
                     (0.5 * (-sigma3 * C_prev_m) * (-sigma3) * (dB3**2 - dt))
            
            dM_mil = (-sigma4 * M_prev_m * dB4) + \
                     (0.5 * (-sigma4 * M_prev_m) * (-sigma4) * (dB4**2 - dt))

            # Suma final
            S_mil[r, i] = max(S_prev_m + dS_drift + dS_mil, 0)
            R_mil[r, i] = max(R_prev_m + dR_drift + dR_mil, 0)
            C_mil[r, i] = max(C_prev_m + dC_drift + dC_mil, 0)
            M_mil[r, i] = max(M_prev_m + dM_drift + dM_mil, 0)

    return S_det, R_det, C_det, M_det, S_mil, R_mil, C_mil, M_mil, t


# --- 4. FUNCIÓN PARA PREPARAR DATAFRAME (Al estilo de tu amigo) ---
def preparar_dataframe(t, data_det, data_mil, n_realizations):
    data_mil_T = data_mil.T 
    
    data = {}
    data["Determinista"] = data_det
    for i in range(n_realizations):
        # Cambiamos la etiqueta a "Milstein"
        data[f"Milstein-{i+1}"] = data_mil_T[:, i]
        
    df = pd.DataFrame(data, index=t)
    df.index.name = "Tiempo"
    return df


# --- 5. EJECUCIÓN Y GRÁFICOS ---
with st.spinner("Ejecutando simulaciones (Milstein)..."):
    
    # 1. Llamamos a la simulación (en caché)
    S_det, R_det, C_det, M_det, S_mil, R_mil, C_mil, M_mil, t = simular_modelos_milstein(
        n=n, iota_a=iota_a, iota_b=iota_b, iota_c=iota_c, iota_h=iota_h,
        iota_g=iota_g, iota_m=iota_m, iota_r=iota_r,
        sigma1=sigma1, sigma2=sigma2, sigma3=sigma3, sigma4=sigma4,
        S0=S0, R0=R0, C0=C0, M0=M0,
        T=T, dt=dt, n_realizations=n_realizations
    )
    
    # 2. Preparamos los DataFrames
    df_S = preparar_dataframe(t, S_det, S_mil, n_realizations)
    df_R = preparar_dataframe(t, R_det, R_mil, n_realizations)
    df_C = preparar_dataframe(t, C_det, C_mil, n_realizations)
    df_M = preparar_dataframe(t, M_det, M_mil, n_realizations)

    # 3. Mostramos los gráficos (en 2 columnas para que sean "grandes")
    col_graf1, col_graf2 = st.columns(2)
    
    with col_graf1:
        st.subheader("📈 dsRNA (S)")
        st.line_chart(df_S)
        
    with col_graf2:
        st.subheader("📈 RISC (R)")
        st.line_chart(df_R)

    with col_graf1:
        st.subheader("📈 Complejo (C)")
        st.line_chart(df_C)
        
    with col_graf2:
        st.subheader("📈 mRNA (M)")
        st.line_chart(df_M)