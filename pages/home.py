import streamlit as st

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