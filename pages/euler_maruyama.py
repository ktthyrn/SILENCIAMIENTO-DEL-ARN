import streamlit as st
import numpy as np
import pandas as pd

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide")

st.title("🧮 Comparación: Determinista vs. Euler-Maruyama")
st.write("""
Utiliza los parámetros de la barra lateral para ajustar el modelo.
Cada gráfico compara el modelo determinista (línea oscura) con 
**una sola realización** estocástica (línea clara) para cada variable.
""")

# --- 2. BARRA LATERAL (CON SLIDERS) ---
st.sidebar.header("⚙️ Parámetros de Simulación (RNAi)")

with st.sidebar.expander("Parámetros del Modelo (ι)", expanded=True):
    col1, col2 = st.columns(2)
    n = col1.slider("n (tasa RISC)", 1, 20, 5, key="em_n")
    iota_a = col1.slider("ι_a", 0.0, 20.0, 10.0, format="%.1f", key="em_ia")
    iota_b = col1.slider("ι_b", 0.0, 0.01, 0.001, format="%.4f", key="em_ib")
    iota_c = col1.slider("ι_c", 0.0, 5.0, 1.0, format="%.1f", key="em_ic")
    iota_h = col2.slider("ι_h", 500.0, 2000.0, 1000.0, key="em_ih")
    iota_g = col2.slider("ι_g", 0.0, 5.0, 1.0, format="%.1f", key="em_ig")
    iota_m = col2.slider("ι_m", 0.0, 5.0, 1.0, format="%.1f", key="em_im")
    iota_r = col2.slider("ι_r", 0.0, 1.0, 0.1, format="%.2f", key="em_ir")

with st.sidebar.expander("Condiciones Iniciales", expanded=True):
    col1, col2 = st.columns(2)
    S0 = col1.slider("S_0 (dsRNA)", 0.0, 100.0, 10.0, key="em_s0")
    R0 = col1.slider("R_0 (RISC)", 0.0, 100.0, 0.0, key="em_r0")
    C0 = col2.slider("C_0 (Complejo)", 0.0, 100.0, 0.0, key="em_c0")
    M0 = col2.slider("M_0 (mRNA)", 500.0, 2000.0, 1000.0, key="em_m0")

with st.sidebar.expander("Parámetros de Simulación", expanded=True):
    # ¡VALOR POR DEFECTO 0.01 (PRECISO)!
    T = st.slider("Tiempo Total (T)", 10.0, 200.0, 50.0, key="em_T")
    dt = st.slider("Paso (dt)", 0.001, 0.1, 0.01, format="%.3f", key="em_dt") # Default 0.01

with st.sidebar.expander("Intensidad de Ruido (σ)", expanded=True):
    col1, col2 = st.columns(2)
    sigma1 = col1.slider("σ_1", 0.0, 1.0, 0.1, format="%.2f", key="em_s1")
    sigma2 = col1.slider("σ_2", 0.0, 1.0, 0.1, format="%.2f", key="em_s2")
    sigma3 = col2.slider("σ_3", 0.0, 1.0, 0.1, format="%.2f", key="em_s3")
    sigma4 = col1.slider("σ_4", 0.0, 1.0, 0.1, format="%.2f", key="em_s4")


# --- 3. FUNCIÓN DE SIMULACIÓN (CORREGIDA) ---
@st.cache_data
def simular_modelos(n, iota_a, iota_b, iota_c, iota_h, iota_g, iota_m, iota_r,
                    sigma1, sigma2, sigma3, sigma4,
                    S0, R0, C0, M0,
                    T, dt):
    
    n_realizations = 1
    N = int(T / dt) + 1 
    t = np.linspace(0, T, N)
    sqrt_dt = np.sqrt(dt)

    S_det, R_det, C_det, M_det = (np.zeros(N) for _ in range(4))
    S_det[0], R_det[0], C_det[0], M_det[0] = S0, R0, C0, M0

    S_em, R_em, C_em, M_em = (np.zeros(N) for _ in range(4))
    S_em[0], R_em[0], C_em[0], M_em[0] = S0, R0, C0, M0

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

        # --- Modelo Estocástico (Euler-Maruyama) ---
        dB1, dB2, dB3, dB4 = [np.random.normal(0, sqrt_dt) for _ in range(4)]
        S_prev_e, R_prev_e, C_prev_e, M_prev_e = S_em[i-1], R_em[i-1], C_em[i-1], M_em[i-1]

        dS_det_e = (-iota_a * S_prev_e + iota_g * C_prev_e) * dt
        dR_det_e = (n * iota_a * S_prev_e - iota_r * R_prev_e - iota_b * R_prev_e * M_prev_e) * dt
        dC_det_e = (iota_b * R_prev_e * M_prev_e - (iota_g + iota_c) * C_prev_e) * dt
        dM_det_e = (iota_h - iota_m * M_prev_e - iota_b * R_prev_e * M_prev_e) * dt

        dS_stoch = -sigma1 * S_prev_e * dB1
        dR_stoch = sigma1 * n * S_prev_e * dB1 - sigma2 * R_prev_e * dB2
        dC_stoch = -sigma3 * C_prev_e * dB3
        dM_stoch = -sigma4 * M_prev_e * dB4

        S_em[i] = max(S_prev_e + dS_det_e + dS_stoch, 0)
        R_em[i] = max(R_prev_e + dR_det_e + dR_stoch, 0)
        C_em[i] = max(C_prev_e + dC_det_e + dC_stoch, 0)
        M_em[i] = max(M_prev_e + dM_det_e + dM_stoch, 0)

    return S_det, R_det, C_det, M_det, S_em, R_em, C_em, M_em, t

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
    S_det, R_det, C_det, M_det, S_em, R_em, C_em, M_em, t = simular_modelos(
        n=n, iota_a=iota_a, iota_b=iota_b, iota_c=iota_c, iota_h=iota_h,
        iota_g=iota_g, iota_m=iota_m, iota_r=iota_r,
        sigma1=sigma1, sigma2=sigma2, sigma3=sigma3, sigma4=sigma4,
        S0=S0, R0=R0, C0=C0, M0=M0,
        T=T, dt=dt
    )
    
    # 2. Preparamos 4 DataFrames SEPARADOS
    df_S = pd.DataFrame({"S (Determinista)": S_det, "S (Euler-Maruyama)": S_em}, index=t)
    df_R = pd.DataFrame({"R (Determinista)": R_det, "R (Euler-Maruyama)": R_em}, index=t)
    df_C = pd.DataFrame({"C (Determinista)": C_det, "C (Euler-Maruyama)": C_em}, index=t)
    df_M = pd.DataFrame({"M (Determinista)": M_det, "M (Euler-Maruyama)": M_em}, index=t)

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