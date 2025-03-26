"""
Prediction logic for the plant disease classification project.
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import logging
from src.model import load_model, load_class_indices
from src.preprocessing import preprocess_image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

MODEL_PATH = "../models/Model1.keras"
PICKLE_PATH = "../models/Model1.pkl"

def predict_image(image_path: str) -> tuple[str, float]:
    """
    Predict the plant disease from an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: Predicted class name and confidence score.

    Raises:
        RuntimeError: If prediction fails.
    """
    try:
        # Load model and class indices
        model = load_model(MODEL_PATH)
        class_indices = load_class_indices(PICKLE_PATH)
        
        # Preprocess image
        image = preprocess_image(image_path)
        
        # Predict
        prediction = model.predict(image, verbose=0)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        
        class_names_dict = {v: k for k, v in class_indices.items()}
        predicted_class_name = class_names_dict[predicted_class_idx]
        
        if predicted_class_name not in class_names:
            raise ValueError(f"Predicted class {predicted_class_name} not in expected class names")
        
        logger.info(f"Model prediction: {predicted_class_name} with confidence {confidence}")
        return predicted_class_name, confidence
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise RuntimeError(f"Prediction error: {str(e)}")