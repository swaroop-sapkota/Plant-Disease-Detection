import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", layout="wide")

# Custom CSS for improved design
st.markdown("""
    <style>
        .main {
            background-color: #f7fff7;
        }
        .block-container {
            padding: 2rem 3rem;
        }
        h1, h2, h3 {
            color: #2e7d32;
        }
        .stButton>button {
            background-color: #2e7d32;
            color: white;
            border-radius: 5px;
            height: 3em;
            width: 100%;
        }
        .stFileUploader label {
            color: #2e7d32;
        }
    </style>
""", unsafe_allow_html=True)

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('Plant_Disease_Dataset/trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    col1, col2 = st.columns(2)
    with col1:
        st.image("Plant_Disease_Dataset/home_page.jpeg", use_container_width=True)
    with col2:
        st.markdown("""
        # 🌱 Plant Disease Recognition System
        Welcome to our intelligent system for detecting plant diseases!

        Upload an image of a crop leaf and let our model do the rest.

        ### Why Use This?
        - ✅ Accurate
        - ⚡ Fast
        - 🧠 AI-powered

        Head over to the **Disease Recognition** tab to begin your diagnosis.
        """)

# About Page
elif app_mode == "About":
    st.markdown("""
    # About This Project
    
    This system is built on a dataset of over 87,000 images of healthy and diseased crop leaves.

    ### Dataset Details:
    - **Train:** 70,295 images
    - **Validation:** 17,572 images
    - **Test:** 33 images

    The model is trained to detect **38** different classes of plant diseases using deep learning.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.markdown("""
    # 🌾 Disease Recognition
    Upload a crop image below to identify potential diseases.
    """)

    uploaded_file = st.file_uploader("Choose a plant image:", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        if st.button("Predict Disease"):
            with st.spinner("Analyzing the image..."):
                result_index = model_prediction(uploaded_file)
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
                    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
                    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

                st.success(f"✅ Prediction: {class_name[result_index]}")
