import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Load your trained model
model = tf.keras.models.load_model('best_model.h5')


# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    image = image.resize((224, 224))               # Resize image
    # image = np.array(image) / 255.0                # Normalize
    image = np.array(image)               # Normalize
    image = np.expand_dims(image, axis=0)          # Add batch dimension
    return image


# Function to predict pneumonia
def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    # print(prediction)
    return prediction

    
    
# Streamlit UI
st.title("Pneumonia Detection from X-Ray")
st.write("Upload an X-ray image to check if the person has Pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])
image_with_text = None
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    # st.image(image, caption='Uploaded X-ray Image', use_container_width=True)
    # Make prediction
    result = predict_image(uploaded_file)
    

    bg_color = "#4CAF50"
    if result[0][0] > 0.5:
        # Display result
        st.write("### The Person has Pneumonia.")
        # st.write("#### "+result)
        st.image(image, caption='📷 Original X-ray', use_container_width=True)
    else:
        # Display result
        st.write("### The Person is Normal.")
        # st.write("#### "+result)
        st.image(image, caption='📷 Original X-ray', use_container_width=True)
        
# Run this Streamlit app using: streamlit run app.py
