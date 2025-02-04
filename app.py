import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('HandwrittenImageRecognition.keras')

st.header('HAND DIGIT RECOGNITION ML MODEL')

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Upload an image of a handwritten digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = np.array(image)

    # Resize to match model input size (assuming 28x28)
    image = cv2.resize(image, (28, 28))

    # Normalize and invert colors
    image = np.invert(image) / 255.0

    # Reshape to fit the model input (batch_size, height, width, channels)
    image = image.reshape(1, 28, 28, 1)

    # Display the uploaded image
    st.image(image.reshape(28, 28), caption="Uploaded Image", width=150)

    # Make prediction
    output = model.predict(image)
    predicted_digit = np.argmax(output)

    # Show result
    st.success(f'Digit in the image is: {predicted_digit}')
