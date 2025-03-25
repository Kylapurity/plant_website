# src/model.py
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import pickle
from src.preprocessing import get_data_generators

def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
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
    train_generator, validation_generator, test_generator = get_data_generators(dataset_path)
    
    # Load class_names from class_indices.pkl (or it will be validated in preprocessing.py)
    with open('class_indices.pkl', 'rb') as f:
        class_indices = pickle.load(f)
    class_names = list(class_indices.keys())

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
    model.save('model.keras')
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('class_indices.pkl', 'wb') as f:
        pickle.dump(train_generator.class_indices, f)
    
    return model, history

def retrain_model(new_dataset_path):
    print("Loading new data for retraining...")
    train_generator, validation_generator, _ = get_data_generators(new_dataset_path)
    
    with open('class_indices.pkl', 'rb') as f:
        class_indices = pickle.load(f)
    class_names = list(class_indices.keys())

    if train_generator.num_classes != len(class_names):
        raise ValueError(f"Expected {len(class_names)} classes, but found {train_generator.num_classes} classes in the new dataset.")
    
    print("Loading existing model...")
    model = load_model('model.keras')
    
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
    model.save('model.keras')
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, history