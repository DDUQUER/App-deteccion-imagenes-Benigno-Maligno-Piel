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

# ID del modelo y del logo en Google Drive
model_file_id = '1FkwEZ3XZ466e9LWcY51Ibl5_9vBtSadI'  # Reemplaza esto con tu ID de archivo real
logo_file_id = '1cDozrnUSUxaSIwpd5q65qAMzna31GMzG'  # ID del logo

# Nombres de archivos a guardar
model_destination = "model_fin_EN0_6931.h5"
logo_destination = "PixelDerm_logo.png"

# Descargar el modelo
download_file_from_google_drive(model_file_id, model_destination)
# Descargar el logo
download_file_from_google_drive(logo_file_id, logo_destination)

# Cargar el modelo de clasificación binaria
model_binary = load_model(model_destination)

# Funciones para procesar imágenes y realizar predicciones
def preprocess_image(img, target_size):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image_binary(img):
    processed_image = preprocess_image(img, (256, 256))
    prediction = model_binary.predict(processed_image)
    return prediction

# Función principal para la aplicación
def main():
    st.title("Aplicación de Detección de Lesiones en la Piel 🧐")

    # Mostrar el logo justo debajo del título
    logo = Image.open(logo_destination)
    st.image(logo, caption="Logo de la aplicación", use_column_width=True)

    # Barra lateral para la navegación
    st.sidebar.title("Navegación")
    menu = st.sidebar.radio("Ir a", ["Home", "Detección Lunar - Benigno Maligno"])

    if menu == "Home":
        show_home()
    elif menu == "Detección Lunar - Benigno Maligno":
        show_detection_benigno_maligno()

# Página principal (Home)
def show_home():
    st.write(
        """
       ### 🌟 Bienvenido a la Aplicación de Predicción de Lesiones Cutáneas 🌟

       Desarrollada como parte de un trabajo de fin de máster, esta aplicación utiliza un modelo entrenado con **10,599 imágenes dermatoscópicas** 🖼️, recopiladas en el conjunto de datos **HAM10000** ("Human Against Machine with 10000 training images"), proporcionado por la **ISIC** (International Skin Imaging Collaboration).

       Este modelo se basa en técnicas avanzadas de **Deep Learning** 🧠 para ayudar a identificar de manera precisa si una lesión cutánea es **benigna** o **maligna** 🔎. Sin embargo, es importante aclarar que **los resultados de esta aplicación no constituyen un diagnóstico médico**. Este sistema utiliza **aprendizaje automático** para generar aproximaciones y predicciones probabilísticas, pero no debe sustituir la evaluación de un profesional de la salud 👩‍⚕️👨‍⚕️.

       ### Clasificación de Lesiones Cutáneas🔎

       La reclasificación de los datos para el entrenamiento del modelo se realizaron de la siguiente manera:

       **Maligno:**
          - AKIEC (Queratinosis Actínica)
          - BCC (Carcinoma de Células Basales)
          - MEL (Melanoma)
          - VASC (Lesiones Vasculares)

       **Benigno:**
          - BKL (Queratosis Benigna)
          - DF (Dermatofibroma)
          - NV (Nevus Melanocítico)

       **Nota 📌:** Las lesiones vasculares se clasificaron como malignas, aunque se sabe que algunas pueden ser benignas. Esta clasificación se realiza con el fin de disminuir los casos de falsos negativos (aquellos que el modelo clasifica como benignos pero que son malignos).

       ✨ Esperamos que esta herramienta te sea de gran utilidad y contribuya a una mejor comprensión de las lesiones cutáneas. ¡Gracias por visitarnos! 🌈
        """
    )

# Página de Detección Lunar - Benigno/Maligno
def show_detection_benigno_maligno():
    st.header("Detección Lunar - Benigno o Maligno 🔬")
    st.write(
        """
        Esta página te permite detectar si una lesión en la piel es benigna o maligna.
        
        Nota 📌: Para mejorar la precisión del modelo, intente que la mancha de piel esté centrada y ocupe la mayor parte de la imagen. No se preocupe si, al hacerlo, la calidad de la imagen se reduce. 
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




