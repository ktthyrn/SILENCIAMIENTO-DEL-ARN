import streamlit as st
import numpy as np
import pandas as pd  # Importante: para manejar los datos del gráfico

# --- 1. CONFIGURACIÓN DE PÁGINA (Para gráficos "grandes") ---
st.set_page_config(layout="wide")

st.title("Comparación: Euler Determinista vs. Euler-Maruyama")


# --- 2. BARRA LATERAL (¡AHORA CON SLIDERS!) ---
st.sidebar.header("⚙️ Parámetros de Simulación (RNAi)")

with st.sidebar.expander("Parámetros del Modelo (ι)", expanded=True):
    # Usamos st.columns para que no se vea tan largo
    col1, col2 = st.columns(2)
    # --- Columna 1 ---
    n = col1.slider("n (tasa RISC)", 1, 20, 5)
    iota_a = col1.slider("ι_a", 0.0, 20.0, 10.0, format="%.1f")
    iota_b = col1.slider("ι_b", 0.0, 0.01, 0.001, format="%.4f")
    iota_c = col1.slider("ι_c", 0.0, 5.0, 1.0, format="%.1f")
    # --- Columna 2 ---
    iota_h = col2.slider("ι_h", 500.0, 2000.0, 1000.0)
    iota_g = col2.slider("ι_g", 0.0, 5.0, 1.0, format="%.1f")
    iota_m = col2.slider("ι_m", 0.0, 5.0, 1.0, format="%.1f")
    iota_r = col2.slider("ι_r", 0.0, 1.0, 0.1, format="%.2f")

with st.sidebar.expander("Condiciones Iniciales", expanded=True):
    col1, col2 = st.columns(2)
    S0 = col1.slider("S_0 (dsRNA)", 0.0, 100.0, 10.0)
    R0 = col1.slider("R_0 (RISC)", 0.0, 100.0, 0.0)
    C0 = col2.slider("C_0 (Complejo)", 0.0, 100.0, 0.0)
    M0 = col2.slider("M_0 (mRNA)", 500.0, 2000.0, 1000.0)

with st.sidebar.expander("Parámetros de Simulación", expanded=True):
    col1, col2 = st.columns(2)
    T = col1.slider("Tiempo Total (T)", 10.0, 200.0, 50.0, key="rnai_T")
    dt = col1.slider("Paso (dt)", 0.001, 0.1, 0.01, format="%.3f", key="rnai_dt")
    n_realizations = col2.slider("Nº Realizaciones (EM)", 1, 20, 5, key="rnai_n_realiz")

with st.sidebar.expander("Intensidad de Ruido (σ)", expanded=True):
    col1, col2 = st.columns(2)
    sigma1 = col1.slider("σ_1", 0.0, 1.0, 0.1, format="%.2f")
    sigma2 = col1.slider("σ_2", 0.0, 1.0, 0.1, format="%.2f")
    sigma3 = col2.slider("σ_3", 0.0, 1.0, 0.1, format="%.2f")
    sigma4 = col2.slider("σ_4", 0.0, 1.0, 0.1, format="%.2f")


# --- 3. FUNCIÓN DE SIMULACIÓN (CON CACHÉ) ---
# (Ahora calcula ambos modelos, como pediste)
@st.cache_data
def simular_modelos(n, iota_a, iota_b, iota_c, iota_h, iota_g, iota_m, iota_r,
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

    # --- Arrays para Euler-Maruyama ---
    S_em, R_em, C_em, M_em = (np.zeros((n_realizations, N)) for _ in range(4))
    for r in range(n_realizations):
        S_em[r, 0], R_em[r, 0], C_em[r, 0], M_em[r, 0] = S0, R0, C0, M0

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

        # --- Modelo Estocástico (Euler-Maruyama) ---
        for r in range(n_realizations):
            dB1, dB2, dB3, dB4 = [np.random.normal(0, sqrt_dt) for _ in range(4)]
            S_prev_e, R_prev_e, C_prev_e, M_prev_e = S_em[r, i-1], R_em[r, i-1], C_em[r, i-1], M_em[r, i-1]

            dS_det_e = (-iota_a * S_prev_e + iota_g * C_prev_e) * dt
            dR_det_e = (n * iota_a * S_prev_e - iota_r * R_prev_e - iota_b * R_prev_e * M_prev_e) * dt
            dC_det_e = (iota_b * R_prev_e * M_prev_e - (iota_g + iota_c) * C_prev_e) * dt
            dM_det_e = (iota_h - iota_m * M_prev_e - iota_b * R_prev_e * M_prev_e) * dt

            dS_stoch = -sigma1 * S_prev_e * dB1
            dR_stoch = sigma1 * n * S_prev_e * dB1 - sigma2 * R_prev_e * dB2
            dC_stoch = -sigma3 * C_prev_e * dB3
            dM_stoch = -sigma4 * M_prev_e * dB4

            S_em[r, i] = max(S_prev_e + dS_det_e + dS_stoch, 0)
            R_em[r, i] = max(R_prev_e + dR_det_e + dR_stoch, 0)
            C_em[r, i] = max(C_prev_e + dC_det_e + dC_stoch, 0)
            M_em[r, i] = max(M_prev_e + dM_det_e + dM_stoch, 0)

    return S_det, R_det, C_det, M_det, S_em, R_em, C_em, M_em, t


# --- 4. FUNCIÓN PARA PREPARAR DATAFRAME (Al estilo de tu amigo) ---
def preparar_dataframe(t, data_det, data_em, n_realizations):
    # Transponemos los datos estocásticos para que coincidan con el tiempo
    # (N, n_realizations)
    data_em_T = data_em.T 
    
    # Creamos un diccionario, tal como hizo tu amigo
    data = {}
    data["Determinista"] = data_det
    for i in range(n_realizations):
        data[f"EM-{i+1}"] = data_em_T[:, i]
        
    # Creamos el DataFrame
    df = pd.DataFrame(data, index=t)
    df.index.name = "Tiempo"
    return df


# --- 5. EJECUCIÓN Y GRÁFICOS ---
with st.spinner("Ejecutando simulaciones..."):
    
    # 1. Llamamos a la simulación (en caché)
    S_det, R_det, C_det, M_det, S_em, R_em, C_em, M_em, t = simular_modelos(
        n=n, iota_a=iota_a, iota_b=iota_b, iota_c=iota_c, iota_h=iota_h,
        iota_g=iota_g, iota_m=iota_m, iota_r=iota_r,
        sigma1=sigma1, sigma2=sigma2, sigma3=sigma3, sigma4=sigma4,
        S0=S0, R0=R0, C0=C0, M0=M0,
        T=T, dt=dt, n_realizations=n_realizations
    )
    
    # 2. Preparamos los DataFrames
    df_S = preparar_dataframe(t, S_det, S_em, n_realizations)
    df_R = preparar_dataframe(t, R_det, R_em, n_realizations)
    df_C = preparar_dataframe(t, C_det, C_em, n_realizations)
    df_M = preparar_dataframe(t, M_det, M_em, n_realizations)

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