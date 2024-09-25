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
model_file_id = '1FkwEZ3XZ466e9LWcY51Ibl5_9vBtSadI' 
logo_file_id = '1cDozrnUSUxaSIwpd5q65qAMzna31GMzG'  

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
    st.title("Clasificador de Lesiones Cutáneas: Benignas - Malignas 🧐")

    # Barra lateral para la navegación
    st.sidebar.image(logo_destination)
    menu = st.sidebar.radio("Ir a", ["Home", "Detección Lunar - Benigno Maligno"])

    if menu == "Home":
        show_home()
    elif menu == "Detección Lunar - Benigno Maligno":
        show_detection_benigno_maligno()

# Página principal (Home)
def show_home():
    st.write(
        """
       ### 🌟 Bienvenido a la Aplicación de Clasificación de Lesiones Cutáneas Benignas - Malignas. 🌟

       Desarrollada como parte de un trabajo de fin de máster, esta aplicación utiliza un modelo entrenado con **10,599 imágenes dermatoscópicas** 🖼️, recopiladas de la **ISIC** (International Skin Imaging Collaboration).

       Este modelo se desarrolló con técnicas avanzadas de **Deep Learning** 🧠 para ayudar a identificar si una lesión cutánea es **benigna** o **maligna** 🔎. Sin embargo, es importante aclarar que **los resultados de esta aplicación no constituyen un diagnóstico médico**. Este sistema utiliza **aprendizaje automático** para generar aproximaciones y predicciones probabilísticas, pero no debe sustituir la evaluación de un profesional de la salud 👩‍⚕️👨‍⚕️.

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

       **Nota 📌:** Las lesiones vasculares (VASC) se clasificaron como malignas, aunque se sabe que algunas pueden ser benignas. Esta clasificación se realiza con el fin de disminuir los casos de falsos negativos (aquellos que el modelo clasifica como benignos pero que son malignos).

       ✨ Esperamos que esta herramienta te sea de gran utilidad y contribuya a una mejor comprensión de las lesiones cutáneas. ¡Gracias por visitarnos! 🌈
        """
    )

# Página de Detección Lunar - Benigno/Maligno
def show_detection_benigno_maligno():
    st.header("Detección Lunar - Benigno o Maligno 🔬")
    st.write(
        """
        **¿Cómo usar el aplicativo? 🤔**
        
        Para evaluar una fotografía de una mancha cutánea, haz clic en el recuadro de abajo 📸, selecciona la imagen que deseas revisar y deja que el modelo te proporcione la predicción. Recibirás como resultado la probabilidad de que la imagen sea maligna 🧪 y, en otra línea, la predicción final del modelo 💡.

        **¿Cómo funciona el modelo? 🔍**
        
        Ten en cuenta que la probabilidad total es del 100% ✅. Por ejemplo, si la probabilidad de que una imagen sea maligna es del 51% ⚠️, eso significa que la probabilidad de que sea benigna es del 49% 👍. En este caso, el modelo determinará que la lesión cutánea es maligna. El modelo fue entrenado con un criterio de elección del 50%; por lo tanto, si la probabilidad de ser maligna es inferior al 50% 🌿, se categorizará la imagen como benigna. Si es superior a este porcentaje, se clasificará como maligna 🚫.
        
        **Notas 📌:**
          - Para mejorar la precisión del modelo, intenta que la mancha de piel esté centrada 🎯 y ocupe la mayor parte de la imagen 🖼️. No te preocupes si, al hacerlo, la calidad de la imagen se reduce 📉.
          - De igual manera, puedes probar subir fotos desde distintos ángulos 📸 o posiciones de la zona afectada de tu piel para validar el resultado 🔄.
          - Recuerda que estos resultados son probabilísticos 🔢 y tienen un margen de error ⚠️, y no reemplazan el diagnóstico final que pueda emitir un profesional de la salud 👩‍⚕️👨‍⚕️.
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


