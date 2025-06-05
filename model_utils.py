# model_utils.py
import os
import tensorflow as tf
import numpy as np
import joblib
import traceback

MODEL_PATH = "models/raga_model.h5"
SCALER_PATH = "models/scaler.pkl"
LABELS_PATH = "models/label_classes.npy"

def load_model_with_fallbacks(model_path):
    strategies = [
        lambda: tf.keras.models.load_model(model_path),
        lambda: tf.keras.models.load_model(
            model_path,
            custom_objects={
                'InputLayer': tf.keras.layers.InputLayer,
                'Functional': tf.keras.models.Model
            }
        ),
        lambda: rebuild_and_load(model_path),
        lambda: load_as_tflite(model_path)
    ]
    
    errors = []
    for strategy in strategies:
        try:
            tf.keras.backend.clear_session()
            model = strategy()
            return model, errors
        except Exception as e:
            errors.append(str(e))
    return None, errors

def rebuild_and_load(model_path):
    input_shape = (13,)  # match MFCC shape
    num_classes = len(np.load(LABELS_PATH, allow_pickle=True))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.load_weights(model_path)
    return model

def load_as_tflite(model_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(model_path))
    tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    return interpreter

def load_model_and_utilities():
    try:
        for path, name in [(MODEL_PATH, "Model"), (SCALER_PATH, "Scaler"), (LABELS_PATH, "Labels")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {name}")
        
        model, errors = load_model_with_fallbacks(MODEL_PATH)
        if model is None:
            raise RuntimeError("Model loading failed:\n" + "\n".join(errors))

        scaler = joblib.load(SCALER_PATH)
        labels = np.load(LABELS_PATH, allow_pickle=True).tolist()

        if not hasattr(model, "optimizer"):
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        return model, scaler, labels
    except Exception as e:
        return None, None, None
