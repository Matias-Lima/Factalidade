import streamlit as st
import pandas as pd
import numpy as np
import streamlit_option_menu
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime
import base64
import logging
import pandas as pd
import numpy as np


def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return None


logging.basicConfig(level=logging.INFO)
plt.style.use('dark_background')

st.set_page_config(
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)
# Insert custom CSS for glowing effect
st.markdown(
    """
    <style>
    .cover-glow {
        width: 100%;
        height: auto;
        padding: 3px;
        box-shadow: 
            0 0 5px #000033,
            0 0 10px #000066,
            0 0 15px #000099,
            0 0 20px #0000CC,
            0 0 25px #0000FF,
            0 0 30px #3333FF,
            0 0 35px #6666FF;
        position: relative;
        z-index: -1;
        border-radius: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Load and display sidebar image
img_path = "img/Ic_fractal.png"
img_base64 = img_to_base64(img_path)

if img_base64:
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )

st.sidebar.divider()

menu = ["Main", "Análise MF-DFA", "Análise R/S", "Comparação de Ativos","Análise MF-ADCCA" ,"Informações"]
mode = st.sidebar.radio("Navegue", menu)

st.sidebar.markdown("---")


# =========================================================================
# Título do app
st.title("Análise Fractal")

# Carregando o conteúdo do arquivo conforme o modo selecionado
if mode == "Main":
    with open("paginas/main_inicial.py", encoding="utf-8") as file:
        exec(file.read())
        
elif mode == "Análise MF-DFA":
    with open("paginas/mfdfa.py", encoding="utf-8") as file:
        exec(file.read())

        
elif mode == "Análise R/S":
    with open("paginas/rs.py", encoding="utf-8") as file:
        exec(file.read())


elif mode == "Comparação de Ativos":
    with open("paginas/compare.py", encoding="utf-8") as file:
        exec(file.read())

        
elif mode == "Análise MF-ADCCA":
    with open("paginas/mf_adcca.py", encoding="utf-8") as file:
        exec(file.read())

        
elif mode == "Informações":
    with open("paginas/info.py", encoding="utf-8") as file:
        exec(file.read())
# ---------------------------------
