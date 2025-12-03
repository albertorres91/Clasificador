import streamlit as st
from ClasificadorEmail import ClasificadorEmail

# Cargar el modelo y el vectorizador

# Título de la aplicación
st.title("Clasificador de Correos Electrónicos")

# Área de texto para ingresar el correo electrónico
correo = st.text_area("Ingrese el correo electrónico:")

boton = st.button("Clasificar")

if boton:
    if correo.strip() == "":
        st.warning("Por favor, ingrese un correo electrónico para clasificar.")
    else:
        clasificador = ClasificadorEmail(correo)
        resultado, probabilidad = clasificador.clasificar(correo)

        if resultado is None:
            st.error("Error en el proceso de clasificación. Por favor, verifique los modelos.")
        else:
            if resultado == 0:
                st.success(f"El correo NO es SPAM. Probabilidad: {probabilidad:.2f}")
            else:
                st.error(f"El correo SI es SPAM. Probabilidad: {probabilidad:.2f}")