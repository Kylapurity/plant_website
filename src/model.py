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

KERAS_PATH = "../models/Model.keras"

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
    """Load class indices (not used since we use CLASS_NAMES)."""
    raise NotImplementedError("Class indices loading not implemented for .keras format")

class MetricsCallback(tf.keras.callbacks.Callback):
    """Custom callback to log metrics at the end of each epoch."""
    def __init__(self, validation_generator):
        super(MetricsCallback, self).__init__()
        self.validation_generator = validation_generator

    def on_epoch_end(self, epoch, logs=None):
        # Get validation data
        y_true, y_pred = [], []
        for images, labels in self.validation_generator:
            preds = self.model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
            if len(y_true) >= self.validation_generator.samples:
                break
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        val_accuracy = logs.get('val_accuracy', 0.0)
        
        # Log metrics
        logger.info(f"Epoch {epoch + 1} Metrics: Accuracy={val_accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

def retrain_model(model: tf.keras.Model, data_dir: str, epochs: int = 5, batch_size: int = 32) -> dict:
    """Retrain the model and return evaluation metrics."""
    try:
        train_generator, validation_generator = preprocess_dataset(data_dir, batch_size=batch_size)
        
        # Define the custom callback for logging metrics
        metrics_callback = MetricsCallback(validation_generator)
        
        # Retrain with the callback
        logger.info("Starting model retraining...")
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            verbose=1,
            callbacks=[metrics_callback]
        )
        
        # Evaluate final metrics
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
        
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        logger.info("Retraining completed successfully")
        return {
            'success': True,
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    except Exception as e:
        logger.error(f"Failed to retrain model: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }