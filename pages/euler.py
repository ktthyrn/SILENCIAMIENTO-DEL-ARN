import streamlit as st
import numpy as np
import pandas as pd

# --- 1. CONFIGURACIÓN DE PÁGINA ---
# ¡La línea clave para que se vea "ancho"!
st.set_page_config(layout="wide")

st.title("🔢 Simulación: Modelo Determinista (Método de Euler)")
st.write("""
Utiliza los parámetros de la barra lateral para ajustar el modelo.
Esta simulación muestra la evolución del sistema **sin ruido** (determinista),
calculada usando el Método de Euler.
""")

# --- 2. BARRA LATERAL (CON SLIDERS) ---
# Usamos 'key' únicas (ej: "det_n") para esta página
st.sidebar.header("⚙️ Parámetros de Simulación")

with st.sidebar.expander("Parámetros del Modelo (ι)", expanded=True):
    col1, col2 = st.columns(2)
    n = col1.slider("n (tasa RISC)", 1, 20, 5, key="det_n")
    iota_a = col1.slider("ι_a", 0.0, 20.0, 10.0, format="%.1f", key="det_ia")
    iota_b = col1.slider("ι_b", 0.0, 0.01, 0.001, format="%.4f", key="det_ib")
    iota_c = col1.slider("ι_c", 0.0, 5.0, 1.0, format="%.1f", key="det_ic")
    iota_h = col2.slider("ι_h", 500.0, 2000.0, 1000.0, key="det_ih")
    iota_g = col2.slider("ι_g", 0.0, 5.0, 1.0, format="%.1f", key="det_ig")
    iota_m = col2.slider("ι_m", 0.0, 5.0, 1.0, format="%.1f", key="det_im")
    iota_r = col2.slider("ι_r", 0.0, 1.0, 0.1, format="%.2f", key="det_ir")

with st.sidebar.expander("Condiciones Iniciales", expanded=True):
    col1, col2 = st.columns(2)
    S0 = col1.slider("S_0 (dsRNA)", 0.0, 100.0, 10.0, key="det_s0")
    R0 = col1.slider("R_0 (RISC)", 0.0, 100.0, 0.0, key="det_r0")
    C0 = col2.slider("C_0 (Complejo)", 0.0, 100.0, 0.0, key="det_c0")
    M0 = col2.slider("M_0 (mRNA)", 500.0, 2000.0, 1000.0, key="det_m0")

with st.sidebar.expander("Parámetros de Simulación", expanded=True):
    # ¡VALOR POR DEFECTO 0.01 (PRECISO)!
    T = st.slider("Tiempo Total (T)", 10.0, 200.0, 50.0, key="det_T")
    dt = st.slider("Paso (dt)", 0.001, 0.1, 0.01, format="%.3f", key="det_dt") # Default 0.01


# --- 3. FUNCIÓN DE SIMULACIÓN (SOLO DETERMINISTA) ---
@st.cache_data
def simular_modelo_determinista(n, iota_a, iota_b, iota_c, iota_h, iota_g, iota_m, iota_r,
                                S0, R0, C0, M0,
                                T, dt):
    
    N = int(T / dt) + 1 
    t = np.linspace(0, T, N)

    S_det, R_det, C_det, M_det = (np.zeros(N) for _ in range(4))
    S_det[0], R_det[0], C_det[0], M_det[0] = S0, R0, C0, M0

    # --- Bucle de Simulación ---
    for i in range(1, N):
        
        S_prev_d, R_prev_d, C_prev_d, M_prev_d = S_det[i-1], R_det[i-1], C_det[i-1], M_det[i-1]
        
        dS_det_val = (-iota_a * S_prev_d + iota_g * C_prev_d) * dt
        dR_det_val = (n * iota_a * S_prev_d - iota_r * R_prev_d - iota_b * R_prev_d * M_prev_d) * dt
        dC_det_val = (iota_b * R_prev_d * M_prev_d - (iota_g + iota_c) * C_prev_d) * dt
        dM_det_val = (iota_h - iota_m * M_prev_d - iota_b * R_prev_d * M_prev_d) * dt
        
        S_det[i] = max(S_prev_d + dS_det_val, 0)
        R_det[i] = max(R_prev_d + dR_det_val, 0)
        C_det[i] = max(C_prev_d + dC_det_val, 0)
        M_det[i] = max(M_prev_d + dM_det_val, 0)

    return S_det, R_det, C_det, M_det, t

# --- 4. FUNCIÓN AUXILIAR PARA "DIEZMAR" ---
def downsample_dataframe(df, max_points=1000):
    """Reduce el número de puntos en un dataframe para graficar rápido."""
    if len(df) > max_points:
        step = len(df) // max_points
        return df.iloc[::step]
    return df

# --- 5. EJECUCIÓN Y GRÁFICOS (SEPARADOS) ---
with st.spinner("Ejecutando simulación determinista..."):
    
    # 1. Llamamos a la simulación (en caché)
    S_det, R_det, C_det, M_det, t = simular_modelo_determinista(
        n=n, iota_a=iota_a, iota_b=iota_b, iota_c=iota_c, iota_h=iota_h,
        iota_g=iota_g, iota_m=iota_m, iota_r=iota_r,
        S0=S0, R0=R0, C0=C0, M0=M0,
        T=T, dt=dt
    )
    
    # 2. Preparamos 4 DataFrames SEPARADOS
    df_S = pd.DataFrame({"S (Determinista)": S_det}, index=t)
    df_R = pd.DataFrame({"R (Determinista)": R_det}, index=t)
    df_C = pd.DataFrame({"C (Determinista)": C_det}, index=t)
    df_M = pd.DataFrame({"M (Determinista)": M_det}, index=t)

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