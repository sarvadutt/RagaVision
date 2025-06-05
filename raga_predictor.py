import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import librosa
from io import BytesIO
from model_utils import load_model_and_utilities

# load_raga_model function

def load_raga_model():
    model, scaler, label_classes = load_model_and_utilities()
    if model is None:
        return None, None, None, ["Model loading failed. Check logs."]
    return model, scaler, label_classes, []

def extract_audio_features(audio_data, sr=None):
    try:
        y, sr = librosa.load(audio_data, sr=sr)
        if len(y) == 0:
            return None
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        return None

def predict_raga_from_audio(file):
    model, scaler, label_classes, errors = load_raga_model()
    if errors:
        return None, 0.0, errors

    features = extract_audio_features(BytesIO(file.getvalue()))
    if features is None:
        return None, 0.0, ["❌ Failed to extract features"]

    features_scaled = scaler.transform([features])
    predictions = model.predict(features_scaled)

    try:
        predicted_index = np.argmax(predictions[0])
        predicted_label = label_classes[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100
        return predicted_label, confidence, []
    except Exception as e:
        return None, 0.0, [f"Prediction error: {e}"]
    finally:
        # Force cleanup after prediction
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
