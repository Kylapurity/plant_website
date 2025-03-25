# src/preprocessing.py
import os
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(dataset_path, img_size=(128, 128), batch_size=64):
    # Load class_names from class_indices.pkl
    with open('Model2.pkl', 'rb') as f:
        class_indices = pickle.load(f)
    class_names = list(class_indices.keys())

    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    
    train_categories = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    invalid_train_categories = [cat for cat in train_categories if cat not in class_names]
    if invalid_train_categories:
        raise ValueError(f"Invalid categories found in training data: {invalid_train_categories}. Expected categories: {class_names}")
    
    test_categories = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    invalid_test_categories = [cat for cat in test_categories if cat not in class_names]
    if invalid_test_categories:
        raise ValueError(f"Invalid categories found in test data: {invalid_test_categories}. Expected categories: {class_names}")

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

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator