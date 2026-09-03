import streamlit as st
import tensorflow as tf
import numpy as np
import json
import gdown
import os
from PIL import Image
# Download model from Google Drive
file_id = "1YIgVvlNYNYbwGaySz7O2SLFq-BdzN3AK"
url = f"https://drive.google.com/uc?id={file_id}"
output = "plant_model.keras"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False, fuzzy=True)
# Load model
model = tf.keras.models.load_model("plant_model.keras")
# Load classes
with open("classes.json") as f:
    class_names = json.load(f)


# Convert keys to int
#class_names = {int(k): v for k, v in class_names.items()}
# UI
st.title("🌱 Plant Disease Detection App")
st.warning("⚠️ Please upload only plant leaf images")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")

    st.image(image, use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    idx_to_class = {v: k for k, v in class_names.items()}
    predicted_label = idx_to_class[predicted_class]

    predicted_label = predicted_label.replace("_", " ")

    st.success(f"🌿 Prediction: {predicted_label}")
    st.info(f"📊 Confidence: {confidence:.2f}%")