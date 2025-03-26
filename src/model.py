# src/model.py
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import pickle
from preprocessing import get_data_generators

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
MODEL_PATH = "../models/Model1.keras"
PICKLE_PATH = "../models/Model1.pkl"

def create_model(num_classes):
    """
    Create a CNN model for plant disease classification.

    Args:
        num_classes (int): Number of classes to classify.

    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),  # Define input_shape here
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    return model

def train_and_save(dataset_path):
    """
    Train the model and save it along with class indices.

    Args:
        dataset_path (str): Path to the dataset directory.

    Returns:
        tuple: (trained model, training history)

    Raises:
        ValueError: If the number of classes in the dataset does not match the expected number.
    """
    train_generator, validation_generator, test_generator = get_data_generators(dataset_path)
    
    # Validate the number of classes
    if train_generator.num_classes != len(class_names):
        raise ValueError(f"Expected {len(class_names)} classes, but found {train_generator.num_classes} classes in the dataset.")
    
    print("Starting model training...")
    model = create_model(train_generator.num_classes)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("\nChecking the model on test data:")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    print(f"Test F1 Score: {f1:.4f}")
    
    print("\nSaving the model...")
    model.save(MODEL_PATH)
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(train_generator.class_indices, f)
    
    return model, history

def retrain_model(new_dataset_path):
    """
    Retrain the existing model with a new dataset.

    Args:
        new_dataset_path (str): Path to the new dataset directory.

    Returns:
        tuple: (retrained model, retraining history)

    Raises:
        ValueError: If the number of classes in the new dataset does not match the expected number.
    """
    print("Loading new data for retraining...")
    train_generator, validation_generator, _ = get_data_generators(new_dataset_path)
    
    # Validate the number of classes
    if train_generator.num_classes != len(class_names):
        raise ValueError(f"Expected {len(class_names)} classes, but found {train_generator.num_classes} classes in the new dataset.")
    
    print("Loading existing model...")
    model = load_model(MODEL_PATH)
    
    print("Retraining model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("Saving retrained model...")
    model.save(MODEL_PATH)
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(train_generator.class_indices, f)
    
    return model, history