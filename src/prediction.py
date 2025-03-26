import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import logging
from fastapi import HTTPException

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the class names for the plant disease dataset
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

# Constants
MODEL_PATH = "Model2.keras"
PICKLE_PATH = "Model2.pkl"
IMG_SIZE = (128, 128)

def load_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Load the trained model.

    Args:
        model_path (str): Path to the model file.

    Returns:
        tf.keras.Model: Loaded model.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise RuntimeError(f"Could not load model: {str(e)}")

def load_class_indices(pickle_path: str = PICKLE_PATH) -> dict:
    """
    Load the class indices mapping.

    Args:
        pickle_path (str): Path to the pickle file containing class indices.

    Returns:
        dict: Mapping of class names to indices.

    Raises:
        RuntimeError: If the class indices cannot be loaded.
    """
    try:
        with open(pickle_path, 'rb') as f:
            class_indices = pickle.load(f)
        # Validate that loaded class indices match expected class_names
        loaded_class_names = list(class_indices.keys())
        if loaded_class_names != class_names:
            logger.error(f"Class names in {pickle_path} do not match expected class_names: {class_names}")
            raise ValueError("Loaded class indices do not match expected class_names.")
        return class_indices
    except Exception as e:
        logger.error(f"Error loading class indices from {pickle_path}: {str(e)}")
        raise RuntimeError(f"Could not load class indices: {str(e)}")

def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Preprocess an image for prediction.

    Args:
        image_data (bytes): Raw image data.

    Returns:
        np.ndarray: Preprocessed image array.

    Raises:
        HTTPException: If image preprocessing fails.
    """
    try:
        image = Image.open(image_data).convert("RGB")
        image = image.resize(IMG_SIZE)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Image preprocessing error")

def predict(model: tf.keras.Model, image: np.ndarray, class_indices: dict) -> str:
    """
    Predict the plant disease from an image.

    Args:
        model (tf.keras.Model): Trained model.
        image (np.ndarray): Preprocessed image array.
        class_indices (dict): Mapping of class names to indices.

    Returns:
        str: Predicted class name.

    Raises:
        HTTPException: If prediction fails.
    """
    try:
        prediction = model.predict(image, verbose=0)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        class_names_dict = {v: k for k, v in class_indices.items()}
        predicted_class_name = class_names_dict[predicted_class_idx]
        
        # Validate the predicted class name
        if predicted_class_name not in class_names:
            raise ValueError(f"Predicted class {predicted_class_name} not in expected class names: {class_names}")
        
        logger.info(f"Model prediction: {predicted_class_name}")
        return predicted_class_name
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction error")