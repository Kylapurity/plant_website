import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import logging
from src.preprocessing import preprocess_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

KERAS_PATH = "../models/Model1.keras"

def load_model(keras_path: str = KERAS_PATH) -> tf.keras.Model:
    """Load the Keras model from a .keras file."""
    try:
        logger.info(f"Attempting to load model from {keras_path} with TensorFlow version {tf.__version__}")
        model = tf.keras.models.load_model(keras_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {keras_path}: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to load model from {keras_path}: {str(e)}")

def load_class_indices(pickle_path: str = KERAS_PATH) -> dict:
    """Load class indices from pickle file (not used since we use CLASS_NAMES)."""
    # This function is not needed since we're using CLASS_NAMES directly
    raise NotImplementedError("Class indices loading not implemented for .keras format")

def retrain_model(model: tf.keras.Model, data_dir: str) -> dict:
    """Retrain the model and return evaluation metrics."""
    try:
        train_generator, validation_generator = preprocess_dataset(data_dir)
        
        # Retrain
        logger.info("Starting model retraining...")
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=5,
            verbose=1
        )
        
        # Evaluate
        logger.info("Evaluating retrained model...")
        val_loss, val_accuracy = model.evaluate(validation_generator)
        
        # Detailed metrics
        y_true, y_pred = [], []
        for images, labels in validation_generator:
            preds = model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
            if len(y_true) >= validation_generator.samples:
                break
        
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        logger.info("Retraining completed successfully")
        return {
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    except Exception as e:
        logger.error(f"Failed to retrain model: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to retrain model: {str(e)}")