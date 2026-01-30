import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ======================
# Page config
# ======================
st.set_page_config(
    page_title="Lemon vs Orange",
    page_icon="🍋",
    layout="wide"
)


# ======================
# Minimalist CSS
# ======================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #FAFAFA;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1100px;
    padding-top: 3rem;
}

h1 {
    font-weight: 500;
    letter-spacing: -1px;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #6B6B6B;
    margin-top: 8px;
    margin-bottom: 50px;
}

.section-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #9A9A9A;
    margin-bottom: 14px;
}

.result {
    font-size: 2.4rem;
    font-weight: 600;
    margin-bottom: 6px;
}

.confidence {
    font-size: 1.05rem;
    color: #6B6B6B;
    margin-bottom: 18px;
}

.divider {
    height: 1px;
    background-color: #EDEDED;
    margin: 28px 0;
}

.label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ======================
# Load model
# ======================
MODEL_PATH = r"C:\Users\KyawNyi\Desktop\ML\code\best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["lemon", "orange"]
THRESHOLD = 0.7


# ======================
# Prediction function
# ======================
def import_and_predict(image_data, model):
    image = image_data.resize((128, 128))
    image = image.convert("RGB")
    image = np.asarray(image).astype(np.float32) / 255.0
    image = image[np.newaxis, ...]
    return model.predict(image, verbose=0)[0]


# ======================
# Header
# ======================
st.markdown("""
<h1>🍋 Lemon <span style="color:#6B6B6B;">vs</span> 🍊 Orange</h1>
<p class="subtitle">
    AI-powered fruit classification with confidence awareness
</p>
""", unsafe_allow_html=True)


# ======================
# Two-column layout
# ======================
left, right = st.columns([1, 1])


# ======================
# LEFT — Upload & preview
# ======================
with left:
    st.markdown('<div class="section-title">Upload image</div>', unsafe_allow_html=True)
    file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if file:
        image = Image.open(file)
        st.image(image, width=420)


# ======================
# RIGHT — Prediction
# ======================
with right:
    if file:
        prediction = import_and_predict(image, model)
        confidence = float(np.max(prediction))
        idx = int(np.argmax(prediction))
        label = class_names[idx]

        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

        if confidence < THRESHOLD:
            st.markdown('<div class="result">Unknown</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="confidence">Confidence: {confidence:.2f}</div>',
                unsafe_allow_html=True
            )
        else:
            color = "#F5C400" if label == "lemon" else "#FF8C42"
            emoji = "🍋" if label == "lemon" else "🍊"

            st.markdown(
                f'<div class="result" style="color:{color};">{emoji} {label.capitalize()}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="confidence">Confidence: {confidence:.2f}</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ======================
        # Confidence breakdown
        # ======================
        st.markdown('<div class="section-title">Confidence breakdown</div>', unsafe_allow_html=True)

        # Lemon
        st.markdown(f"""
        <div class="label-row">
            <span>Lemon</span>
            <span>{prediction[0]:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(prediction[0]))

        # Orange
        st.markdown(f"""
        <div class="label-row">
            <span>Orange</span>
            <span>{prediction[1]:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(prediction[1]))

    else:
        st.markdown(
            '<div class="confidence">Upload an image to see the prediction.</div>',
            unsafe_allow_html=True
        )
