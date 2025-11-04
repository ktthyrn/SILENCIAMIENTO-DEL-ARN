import streamlit as st
# Importamos la nueva librería que instalamos
# from streamlit_option_menu import option_menu

# --- 1. CONFIGURACIÓN DE PÁGINA ---

pagina_mov_browniano = st.Page("pages/mov_browniano.py", title="Movimiento Browniano")
pagina_tiempo_explosion = st.Page("pages/tiempo_explosion.py", title="Tiempo de explosión")
pagina_euler = st.Page("pages/euler.py", title="Método de Euler")
pagina_euler_maruyama = st.Page("pages/euler_maruyama.py", title="Método de Euler-Maruyama")
pagina_milstein = st.Page("pages/milstein.py", title="Método de Milstein")

pg = st.navigation([pagina_mov_browniano,
                    pagina_tiempo_explosion,
                    pagina_euler,
                    pagina_euler_maruyama,
                    pagina_milstein])

st.set_page_config(
    page_title="UNMSM - Proyectos de Bioinformática",
    page_icon="🔬", # Un ícono para la pestaña del navegador
    layout="centered", # Usamos "centered" para que no sea tan ancho
    initial_sidebar_state="auto" # "auto" hace que se oculte en móviles
)

pg.run()

st.title("🧬 Silenciamiento del ARN")

    # 1. TÍTULO DE LA SECCIÓN (Como pediste: primero el qué es)
st.header("¿Qué es el Silenciamiento del ARN (RNAi)?")

# 2. TEXTO (Como pediste: luego el texto)
st.write("""
El silenciamiento del ARN, también conocido como interferencia por ARN (ARNi) 
es un mecanismo biológico fundamental conservado en la mayoria de eucariotas. 
Su función principal es la supresión de la expresión de genes específicos a 
nivel postranscripcional...
""")
st.write("""
La función más destacada del silenciamiento del ARN es la discriminación 
entre lo 'propio' (self) y lo 'ajeno' (non-self) a nivel genético. Actúa 
para suprimir la expresión de 'genes ajenos', elementos potencialmente 
dañinos como los codificados en virus o transposones.
""")

# 3. IMAGEN (Como pediste: luego la imagen)
# Streamlit buscará el archivo "diagrama.png" en la misma carpeta
st.image(
    "diagrama.png", 
    caption="Diagrama del mecanismo de Silenciamiento del ARN."
)

# 4. NOMBRES (Como pediste: al final los nombres)
st.markdown("---") # Una línea divisoria
st.subheader("Presentado por:")
st.markdown("""
* Cárdenas Garcia, Katherin Paola
* Carrillo Montero, Julio André
* Limaymanta Curo, Jason
""")

# --- 2. BARRA LATERAL (SIDEBAR) CON EL NUEVO MENÚ ---
# with st.sidebar:
#     # st.sidebar.title("Navegación") # Ya no necesitamos un título feo
    
#     # Aquí creamos el menú profesional.
#     # Es el reemplazo del "st.sidebar.radio"
#     seccion_seleccionada = option_menu(
#         menu_title="Menú Principal",  # Título del menú
#         options=[
#             "Silenciamiento del ARN", 
#             "Movimiento Browniano", 
#             "Tiempo de Explosión", 
#             "Método de Euler", 
#             "Euler-Maruyama", 
#             "Método de Milstein"
#         ],
#         # Aquí puedes buscar íconos: https://icons.getbootstrap.com/
#         icons=[
#             "journal-text", # Un ícono como de "paper" o "artículo"
#             "arrows-move",  # Ícono para movimiento
#             "hourglass-split", # Ícono de tiempo
#             "calculator",   # Ícono de calculadora
#             "graph-up-arrow", # Ícono de gráfico
#             "diagram-3"     # Ícono de diagrama
#         ],
#         menu_icon="cast", # Ícono del menú (opcional)
#         default_index=0,  # Para que empiece en la primera opción
#     )

# # --- 3. CONTENIDO PRINCIPAL (BASADO EN LA SELECCIÓN DEL MENÚ) ---

# # --- SECCIÓN: SILENCIAMIENTO DEL ARN ---
# if seccion_seleccionada == "Silenciamiento del ARN":
    
#     st.title("🧬 Silenciamiento del ARN")

#     # 1. TÍTULO DE LA SECCIÓN (Como pediste: primero el qué es)
#     st.header("¿Qué es el Silenciamiento del ARN (RNAi)?")
    
#     # 2. TEXTO (Como pediste: luego el texto)
#     st.write("""
#     El silenciamiento del ARN, también conocido como interferencia por ARN (ARNi) 
#     es un mecanismo biológico fundamental conservado en la mayoria de eucariotas. 
#     Su función principal es la supresión de la expresión de genes específicos a 
#     nivel postranscripcional...
#     """)
#     st.write("""
#     La función más destacada del silenciamiento del ARN es la discriminación 
#     entre lo 'propio' (self) y lo 'ajeno' (non-self) a nivel genético. Actúa 
#     para suprimir la expresión de 'genes ajenos', elementos potencialmente 
#     dañinos como los codificados en virus o transposones.
#     """)
    
#     # 3. IMAGEN (Como pediste: luego la imagen)
#     # Streamlit buscará el archivo "diagrama.png" en la misma carpeta
#     st.image(
#         "diagrama.png", 
#         caption="Diagrama del mecanismo de Silenciamiento del ARN."
#     )
    
#     # 4. NOMBRES (Como pediste: al final los nombres)
#     st.markdown("---") # Una línea divisoria
#     st.subheader("Presentado por:")
#     st.markdown("""
#     * Cárdenas Garcia, Katherin Paola
#     * Carrillo Montero, Julio André
#     * Limaymanta Curo, Jason
#     """)

# # --- SECCIÓN: MOVIMIENTO BROWNIANO ---
# elif seccion_seleccionada == "Movimiento Browniano":
#     st.title("🚶‍♂️ Movimiento Browniano")
#     st.write("Aquí irá el contenido y las simulaciones sobre el Movimiento Browniano.")
#     # Puedes añadir gráficos, sliders, etc.

# # --- SECCIÓN: TIEMPO DE EXPLOSIÓN ---
# elif seccion_seleccionada == "Tiempo de Explosión":
#     st.title("⏱️ Tiempo de Explosión")
#     st.write("Esta sección explorará modelos relacionados con el tiempo de explosión.")

# # --- SECCIÓN: MÉTODO DE EULER ---
# elif seccion_seleccionada == "Método de Euler":
#     st.title("🔢 Método de Euler")
#     st.write("Detalles y ejemplos de la aplicación del Método de Euler.")

# # --- SECCIÓN: EULER-MARUYAMA ---
# elif seccion_seleccionada == "Método de Euler-Maruyama":
#     st.title("🧮 Método de Euler-Maruyama")
#     st.write("Aquí se presentará el Método de Euler-Maruyama para Ecuaciones Diferenciales Estocásticas (EDEs).")

# # --- SECCIÓN: MÉTODO DE MILSTEIN ---
# elif seccion_seleccionada == "Método de Milstein":
#     st.title("🔬 Método de Milstein")
#     st.write("Exploración del Método de Milstein.")