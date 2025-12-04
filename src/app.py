import streamlit as st
from ClasificadorEmail import ClasificadorEmail

# Configuración de página
st.set_page_config(page_title="Clasificador de Correos", layout="wide")

# Título principal
st.title("Clasificador de Correos Electrónicos")

# Input del usuario
correo = st.text_area("Ingrese el correo electrónico:", height=200)

# Botón de acción
if st.button("Clasificar"):

    if correo.strip() == "":
        st.warning("Por favor, ingrese un correo electrónico para clasificar.")
    else:
        clasificador = ClasificadorEmail(correo)
        resultado, probabilidad = clasificador.clasificar(correo)

        if resultado is None:
            st.error("Error en el proceso de clasificación. Por favor, verifique los modelos.")
        else:
            # Si por alguna razón no hay probabilidad, la mostramos sin formato
            if probabilidad is None:
                if resultado == "spam":
                    st.error("El correo SÍ es SPAM.")
                else:
                    st.success("El correo NO es SPAM.")
            else:
                if resultado == "spam":
                    st.error(f"El correo SÍ es SPAM. Probabilidad: {probabilidad:.2f}")
                else:
                    st.success(f"El correo NO es SPAM. Probabilidad: {probabilidad:.2f}")
