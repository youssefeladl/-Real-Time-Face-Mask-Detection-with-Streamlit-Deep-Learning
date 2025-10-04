import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

# === Load model ===
model = load_model(r"D:\materials\AI track\materials\DEEP LEARNING\CV\face_mask_detection\pre_trained_model.keras")  
img_size = 128  

# === Page settings ===
st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

# === Custom Style ===
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .title {
        font-size: 42px;
        color: #0d47a1;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub {
        font-size: 18px;
        text-align: center;
        color: #555;
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 20px;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)

# === Title ===
st.markdown('<div class="title">😷 Face Mask Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload or capture your photo to check if you are wearing a mask.</div>', unsafe_allow_html=True)

# === Upload or Camera ===
option = st.radio("Choose input method:", ["📁 Upload Image", "📷 Use Camera"])
uploaded_image = None

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
elif option == "📷 Use Camera":
    uploaded_image = st.camera_input("Take a photo")
    if uploaded_image is not None:
        uploaded_image = Image.open(uploaded_image)

# === Prediction Logic ===
def predict(img_pil):
    img = img_pil.convert('RGB') 
    img = img.resize((img_size, img_size))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) 
    prediction = model.predict(img_array)[0][0]
    return prediction

# === Display result ===
if uploaded_image:
    st.image(uploaded_image, caption="Input Image", width=250)
    with st.spinner("Analyzing..."):
        time.sleep(2)
        try:
            result = predict(uploaded_image)
            if result >= 0.5:
                st.error("❌ **No Mask Detected!** 😷")
            else:
                st.success("✅ **Mask Detected!** 👏")
        except Exception as e:
            st.warning(f"⚠️ Error during prediction: {e}")

# === Footer ===
st.markdown("""
    <div class="footer">
    🚀 Deployed by <strong style="color:#0d47a1;">AI Engineer Youssef Eladl</strong>
    </div>
""", unsafe_allow_html=True)
