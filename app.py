import os
import time
import requests
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ========= Config =========
LOCAL_MODEL_PATH = "models/pre_trained_model.keras"
REMOTE_MODEL_URL = ""  # optional direct download link for large model file
IMG_SIZE = 128

st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

# ========= Helpers =========
@st.cache_resource(show_spinner=False)
def download_model_if_needed() -> str:
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    if not REMOTE_MODEL_URL:
        raise FileNotFoundError(
            "Model file not found locally and REMOTE_MODEL_URL is empty. "
            "Add models/pre_trained_model.keras to the repo (<100MB) or provide a direct download link."
        )
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
    tmp_path = LOCAL_MODEL_PATH + ".part"
    with requests.get(REMOTE_MODEL_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    os.replace(tmp_path, LOCAL_MODEL_PATH)
    return LOCAL_MODEL_PATH

@st.cache_resource(show_spinner=True)
def load_mask_model():
    model_path = download_model_if_needed()
    model = load_model(model_path)
    return model

def predict(model, img_pil):
    img = img_pil.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    return pred

# ========= Styles =========
st.markdown("""
    <style>
    .main { background-color: #f7f9fc; }
    .title { font-size: 42px; color: #0d47a1; text-align: center; font-weight: 800; margin: 8px 0; }
    .sub { font-size: 18px; text-align: center; color: #555; margin-bottom: 24px; }
    .footer { text-align: center; margin-top: 36px; font-size: 16px; color: #888; }
    </style>
""", unsafe_allow_html=True)

# ========= UI =========
st.markdown('<div class="title">😷 Face Mask Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload or capture a photo to check if you are wearing a mask.</div>', unsafe_allow_html=True)

option = st.radio("Choose input method:", ["📁 Upload Image", "📷 Use Camera"])
uploaded_image = None

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
else:
    shot = st.camera_input("Take a photo")
    if shot is not None:
        uploaded_image = Image.open(shot)

# ========= Run =========
if uploaded_image is not None:
    st.image(uploaded_image, caption="Input Image", width=280)
    with st.spinner("Loading model & analyzing..."):
        try:
            model = load_mask_model()
            score = predict(model, uploaded_image)
            time.sleep(0.8)

            if score >= 0.5:
                st.error("❌ **No Mask Detected!** 😷")
            else:
                st.success("✅ **Mask Detected!** 👏")

        except Exception as e:
            st.warning(f"⚠️ Error during prediction: {e}")

st.markdown("""
    <div class="footer">🚀 Deployed by <strong style="color:#0d47a1;">AI Engineer Youssef Eladl</strong></div>
""", unsafe_allow_html=True)
