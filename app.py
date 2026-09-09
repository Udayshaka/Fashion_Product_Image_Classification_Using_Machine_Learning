import streamlit as st
import numpy as np
import joblib
from PIL import Image


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Fashion Image Classifier",
    page_icon="👕",
    layout="centered"
)


# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load("fashion_RF_model.pkl")


model = load_model()


# -----------------------------
# Application Title
# -----------------------------

st.title("👕 Fashion Product Image Classifier")

st.write(
    "Upload a fashion product image and the model will "
    "predict its product category."
)


# -----------------------------
# File Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Display original image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Same preprocessing used during training
    image = image.convert("RGB")
    image = image.resize((30, 40))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize pixel values
    image_array = image_array / 255.0

    # Flatten image
    image_features = image_array.flatten()

    # Convert to 2D array for model
    image_features = image_features.reshape(1, -1)

    # Prediction
    prediction = model.predict(image_features)[0]

    # Display result
    st.success(
        f"Predicted Category: {prediction}"
    )