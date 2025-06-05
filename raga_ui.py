import streamlit as st
import os
from raga_predictor import predict_raga_from_audio
from io import BytesIO
import matplotlib.pyplot as plt
import librosa.display
import librosa
import base64

st.set_page_config(page_title="Raga Predictor", layout="centered")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_path = "static/Background_Ra_ui.png"  # Your image path here
encoded_image = get_base64_of_bin_file(image_path)

st.markdown(
    f"""
    <style>
    /* Background image */
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: relative;
        z-index: 0;
    }}

    /* Overlay to dim the background */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5); /* Black with 50% opacity */
        z-index: -1; /* behind content */
    }}

    /* Make text more readable */
    .css-18e3th9 {
        position: relative;
        z-index: 1;
        color: white; /* text color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎶 Raga Prediction Module")
audio_file = st.file_uploader("Upload your audio file (.mp3/.wav)", type=["mp3", "wav"])

if audio_file:
    predicted_raga, confidence, errors = predict_raga_from_audio(audio_file)

    if errors:
        st.error("Error: " + ", ".join(errors))
    else:
        st.success(f"Predicted Raga: {predicted_raga} ({confidence:.2f}%)")

        y, sr = librosa.load(BytesIO(audio_file.getvalue()), sr=None)
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set(title='Waveform')
        st.pyplot(fig)

        os.makedirs("data", exist_ok=True)
        with open("data/raga_output.txt", "w") as f:
            f.write(predicted_raga)

        if st.button("✅ Proceed"):
            st.success("Prediction saved. You can now return to the main app.")
            st.stop()