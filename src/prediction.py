"""
Prediction logic for the plant disease classification project.
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import logging
from src.preprocessing import preprocess_image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def predict_image(image_path: str, model: tf.keras.Model, class_names: list) -> tuple[str, float]:
    """
    Predict the plant disease from an image.

    Args:
        image_path (str): Path to the image file.
        model (tf.keras.Model): The trained Keras model.
        class_names (list): List of class names in the order of model output indices.

    Returns:
        tuple: Predicted class name and confidence score.

    Raises:
        RuntimeError: If prediction fails.
    """
    try:
        # Preprocess image
        image = preprocess_image(image_path)
        
        # Predict
        prediction = model.predict(image, verbose=0)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        
        # Map index to class name
        if predicted_class_idx >= len(class_names):
            raise ValueError(f"Predicted class index {predicted_class_idx} out of range for class_names")
        predicted_class_name = class_names[predicted_class_idx]
        
        logger.info(f"Model prediction: {predicted_class_name} with confidence {confidence}")
        return predicted_class_name, confidence
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise RuntimeError(f"Prediction error: {str(e)}")