import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pickle
from src.preprocessing import preprocess_dataset

PICKLE_PATH = "../models/Model1.pkl"

def load_model(pickle_path: str = PICKLE_PATH) -> tf.keras.Model:
    """Load the Keras model from a pickle file."""
    try:
        with open(pickle_path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, tf.keras.Model):
            raise ValueError(f"Expected a Keras model in {pickle_path}, got {type(model)}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {pickle_path}: {str(e)}")

def retrain_model(model: tf.keras.Model, data_dir: str) -> dict:
    """Retrain the model and return evaluation metrics."""
    try:
        train_generator, validation_generator = preprocess_dataset(data_dir)
        
        # Retrain
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=5,
            verbose=1
        )
        
        # Evaluate
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
        
        return {
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    except Exception as e:
        raise RuntimeError(f"Failed to retrain model: {str(e)}")