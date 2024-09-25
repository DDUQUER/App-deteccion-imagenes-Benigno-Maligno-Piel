import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import requests

# Función para descargar archivo desde Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('GD') or key.startswith('GAPS'):
            token = value
            break
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
    else:
        response = session.get(URL, params={'id': file_id}, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768): 
            if chunk:
                f.write(chunk)

# ID del archivo en Google Drive (parte del enlace)
file_id = '1FkwEZ3XZ466e9LWcY51Ibl5_9vBtSadI'  # Reemplaza esto con tu ID de archivo real

# Nombre del archivo a guardar
destination = "model_fin_EN0_6931.h5"

# Descargar el archivo
download_file_from_google_drive(file_id, destination)

# Cargar el modelo de clasificación binaria
model_binary = load_model(destination)  # Modelo binario (Benigno/Maligno)

# Función para cargar y procesar la imagen según el modelo binario
def preprocess_image(img, target_size):
    img = img.resize(target_size)  # Ajustar el tamaño de la imagen según el modelo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión extra para el lote (batch)
    img_array = preprocess_input(img_array)  # Normalizar la imagen
    return img_array

# Función para realizar la predicción con el modelo binario
def predict_image_binary(img):
    processed_image = preprocess_image(img, (256, 256))  # Tamaño 256x256 para el modelo binario
    prediction = model_binary.predict(processed_image)
    return prediction

# Función principal para la aplicación
def main():
    st.title("Aplicación de Detección de Lesiones en la Piel 🧐")

    # Barra lateral para la navegación
    st.sidebar.title("Navegación")
    menu = st.sidebar.radio("Ir a", ["Home", "Detección Lunar - Benigno Maligno"])

    # Condicionales para cada página
    if menu == "Home":
        show_home()
    elif menu == "Detección Lunar - Benigno Maligno":
        show_detection_benigno_maligno()

# Página principal (Home)
def show_home():
    st.write(
        """
        Bienvenido a la aplicación de predicción de lesiones cutáneas, desarrollada como parte de un trabajo de fin de máster. 
        Este modelo fue entrenado utilizando técnicas avanzadas de Deep Learning para ayudar a identificar de manera precisa si una lesión cutánea es benigna o maligna.

        Como base de datos principal, se empleó el conjunto de datos HAM10000 ("Human Against Machine with 10000 training images"), 
        recopilado por la ISIC (International Skin Imaging Collaboration). Se recopilaron 10,599 imágenes dermatoscópicas clasificadas en 7 tipos de lesiones cutáneas.

        En el proceso de modelado, las clases originales del dataset fueron recategorizadas en dos grandes grupos: benigno y maligno, de la siguiente forma:

        - **AKIEC (Queratinosis Actínica)**: Maligno.
        - **BCC (Carcinoma de Células Basales)**: Maligno.
        - **BKL (Queratosis Benigna)**: Benigno.
        - **DF (Dermatofibroma)**: Benigno.
        - **MEL (Melanoma)**: Maligno.
        - **NV (Nevus Melanocítico)**: Benigno.
        - **VASC (Lesiones Vasculares)**: Maligno (para minimizar el riesgo de falsos negativos).

        Esperamos que esta herramienta sea de utilidad.
        """
    )

# Página de Detección Lunar - Benigno/Maligno
def show_detection_benigno_maligno():
    st.header("Detección Lunar - Benigno o Maligno 🔬")
    st.write(
        """
        Esta página te permite detectar si una lesión en la piel es benigna o maligna.
        
        Nota📌: Para mejorar la precisión del modelo, intente que la mancha de piel esté centrada y ocupe la mayor parte de la imagen. No se preocupe si, al hacerlo, la calidad de la imagen se reduce. 
        """)

    # Subida de la imagen
    uploaded_image = st.file_uploader("Sube una imagen para la detección", type=["jpg", "png"])
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Imagen cargada", use_column_width=True)

        # Realizar la predicción
        prediction = predict_image_binary(img)
        st.write(f"Predicción: {prediction[0][0] * 100:.2f}% de probabilidad de ser maligno")

        # Mostrar el resultado en formato benigno/maligno
        if prediction[0][0] >= 0.5:
            st.write("El modelo predice que la lesión es **Maligna**.")
        else:
            st.write("El modelo predice que la lesión es **Benigna**.")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()


