# src/preprocessing.py
"""
Data preprocessing for the plant disease classification project.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
IMG_SIZE = (128, 128)
BATCH_SIZE = 64

def get_data_generators(dataset_path: str, img_size: tuple = IMG_SIZE, batch_size: int = BATCH_SIZE):
    """
    Create data generators for training, validation, and testing.

    Args:
        dataset_path (str): Path to the dataset directory with 'train' and 'test' subfolders.
        img_size (tuple): Target image size (height, width).
        batch_size (int): Batch size for the data generators.

    Returns:
        tuple: (train_generator, validation_generator, test_generator)

    Raises:
        ValueError: If the dataset structure does not match the expected class names.
    """
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')

    # Validate dataset structure
    train_categories = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    invalid_train_categories = [cat for cat in train_categories if cat not in class_names]
    if invalid_train_categories:
        raise ValueError(f"Invalid categories found in training data: {invalid_train_categories}. Expected categories: {class_names}")

    test_categories = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    invalid_test_categories = [cat for cat in test_categories if cat not in class_names]
    if invalid_test_categories:
        raise ValueError(f"Invalid categories found in test data: {invalid_test_categories}. Expected categories: {class_names}")

    # Data augmentation for training and validation
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # No augmentation for test data, only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator