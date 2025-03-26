import tensorflow as tf
from PIL import Image
import numpy as np

IMG_SIZE = (128, 128)

def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess a single image for prediction."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def preprocess_dataset(data_dir: str):
    """Preprocess dataset for retraining."""
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, validation_generator