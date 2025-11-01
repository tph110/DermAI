import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np

# Load model
@st.cache_resource
def load_model():
    model = torch.load("model.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

st.title("DermAI - Skin Lesion Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array for OpenCV processing if needed
    img_array = np.array(image)

    # Example: preprocess for your model
    # img_tensor = your_preprocessing_function(img_array)
    # prediction = model(img_tensor)
    
    st.write("Model loaded successfully. You can now add preprocessing and prediction code.")
