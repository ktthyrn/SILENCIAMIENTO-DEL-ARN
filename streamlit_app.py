import streamlit as st

# --- Configuración de la Página ---
# Esto debe ser lo primero que uses de streamlit
st.set_page_config(
    page_title="Silenciamiento del ARN",
    page_icon="🧬"
)

# --- Título y Encabezado ---
st.title("🧬 Entendiendo el Silenciamiento del ARN")
st.write("Esta app explica los conceptos básicos del RNAi (Interferencia de ARN).")
"Hola"
# --- Contenido ---
st.header("¿Qué es el Silenciamiento del ARN (RNAi)?")
st.write("""
El silenciamiento del ARN, o RNAi, es un proceso biológico natural 
en el cual moléculas de ARN inhiben la expresión de genes específicos.
Es un mecanismo clave de defensa celular.
""")

# --- Componente Interactivo ---
st.subheader("Componentes Clave")

# Usamos st.selectbox para crear un menú desplegable
opcion = st.selectbox(
    "Elige un componente para saber más:",
    ("Selecciona uno", "ARN de doble cadena (dsRNA)", "Dicer", "Complejo RISC", "ARNm (ARN mensajero)")
)

# Respondemos a la selección del usuario
if opcion == "ARN de doble cadena (dsRNA)":
    st.write("Es la molécula 'desencadenante'. A menudo proviene de virus o se introduce artificialmente.")
elif opcion == "Dicer":
    st.write("Es una enzima que 'corta' el dsRNA largo en pedazos más pequeños llamados siRNA (ARN de interferencia pequeño).")
elif opcion == "Complejo RISC":
    st.write("Un complejo de proteínas que se une al siRNA. Una hebra del siRNA guía a RISC para encontrar el ARNm objetivo.")
elif opcion == "ARNm (ARN mensajero)":
    st.write("Es el 'mensaje' que lleva las instrucciones del ADN al ribosoma para construir una proteína. RISC destruye este mensaje, 'silenciando' el gen.")