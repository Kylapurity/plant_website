# src/prediction.py
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def load_model(model_path: str):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise RuntimeError(f"Could not load model: {str(e)}")

def load_class_indices(class_indices_path: str):
    try:
        with open(class_indices_path, 'rb') as f:
            class_indices = pickle.load(f)
        return class_indices
    except Exception as e:
        logger.error(f"Error loading class indices from {class_indices_path}: {str(e)}")
        raise RuntimeError(f"Could not load class indices: {str(e)}")

def preprocess_image(image_data: bytes) -> np.ndarray:
    try:
        image = Image.open(image_data).convert("RGB")
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Image preprocessing error")

def predict(model, image, class_indices) -> str:
    try:
        prediction = model.predict(image)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        class_names = list(class_indices.keys())  # Derive class_names from class_indices
        class_names_dict = {v: k for k, v in class_indices.items()}
        predicted_class_name = class_names_dict[predicted_class_idx]
        if predicted_class_name not in class_names:
            raise ValueError(f"Predicted class {predicted_class_name} not in expected class names: {class_names}")
        logger.info(f"Model prediction: {predicted_class_name}")
        return predicted_class_name
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction error")