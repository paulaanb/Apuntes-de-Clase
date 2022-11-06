import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pickle

st.set_page_config(layout='centered', page_icon='👩‍⚕️', page_title='Cancer Detection App')

st.title('Aplicación Realizada por Alumnos de la UAX para la detección temprana de Cancer de Mama')

st.image(Image.open('src/images/cancer.png'))
st.sidebar.image(Image.open('src/images/uax.png'))
st.subheader('A continuación introduce los siguiente datos para que la aplicación pueda realizar la predicción:')

radio = float(st.text_input('Radio', 0))
simetria = float(st.text_input('Simetria', 0))
compacticidad = float(st.text_input('Compacticidad', 0))
textura = float(st.text_input('Textura', 0))

data = {'mean radius': radio,
        'mean symmetry': simetria,
        'mean compactness': compacticidad,
        'mean texture': textura}

df = pd.DataFrame(data, index=[0])

st.subheader('Compruebe que los datos introducidos son correctos')

st.table(df)

enviar = st.button('Enviar datos')

if enviar:

    mm = pickle.load(open('src/scaler.pkl', 'rb'))
    lr = pickle.load(open('src/logisticregression.pkl', 'rb'))

    df = mm.transform(df)
    pred = lr.predict(df)

    if pred[0] == 1:
        st.title('''Buenas noticas
                    Con un 90.6 % de probabilidad podemos afirmar que el tumor es benigno''')
    else:
        st.title('''Sentimos comuniarle con un 90.6 % de probabilidad que el tumor puede ser maligno 
                    Pida cita con su médico para recibir tratamiento lo antes posible''')