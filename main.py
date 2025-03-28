import os
import shutil
import zipfile
import io
import json
import warnings
from datetime import datetime
from typing import List

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ================== INITIAL SETUP ================== #
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')

warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

# ================== CONFIGURATION ================== #
DATABASE_URL = os.getenv("DATABASE_URL").replace("mysql://", "mysql+mysqlconnector://", 1)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

UPLOAD_FOLDER = "uploaded_data"
VISUALIZATION_DIR = "visualizations"
MODEL_DIR = "models"
KERAS_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'zip'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# ================== DATABASE MODELS ================== #
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    predicted_disease = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Retraining(Base):
    __tablename__ = "retrainings"
    id = Column(Integer, primary_key=True, index=True)
    num_classes = Column(Integer, nullable=False)
    training_accuracy = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    class_metrics = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ================== CLASS NAMES ================== #
CLASS_NAMES = [
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

# ================== MODEL LOADING ================== #
try:
    if not os.path.exists(KERAS_PATH):
        raise FileNotFoundError(f"Model file not found at {KERAS_PATH}")
    model = tf.keras.models.load_model(KERAS_PATH)
except Exception as e:
    raise RuntimeError(f"Model initialization failed: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Classifier API",
    description="API for classifying plant diseases from leaf images",
    version="1.0.0"
)

app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def preprocess_image(img_bytes: bytes):
    img = image.load_img(io.BytesIO(img_bytes), target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def save_upload_file(file: UploadFile, destination: str) -> None:
    try:
        with open(destination, "wb") as buffer:
            while chunk := await file.read(8192):
                buffer.write(chunk)
    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        raise

def save_visualizations(y_true, y_pred_classes, target_names):
    class_report = classification_report(y_true, y_pred_classes, target_names=target_names)
    plt.figure(figsize=(10, len(target_names) * 0.5 + 2))
    plt.text(0.01, 0.99, class_report, {'fontsize': 10}, fontfamily='monospace')
    plt.axis('off')
    plt.title("Classification Report")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "classification_report.png"), bbox_inches='tight', dpi=300)
    plt.close()

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(max(10, len(target_names)), max(10, len(target_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "confusion_matrix.png"), bbox_inches='tight', dpi=300)
    plt.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Plant Disease Classifier API",
        "status": "operational",
        "model_status": "loaded" if model else "not loaded",
        "class_count": len(CLASS_NAMES),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=dict, tags=["Prediction"])
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        start_time = datetime.now()
        
        if not file.filename or not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        await save_upload_file(file, filepath)
        
        img_bytes = open(filepath, 'rb').read()
        img = preprocess_image(img_bytes)
        predictions = model.predict(img)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        
        prediction = Prediction(
            predicted_disease=predicted_class,
            confidence=float(confidence)
        )
        db.add(prediction)
        db.commit()
        
        return {
            "filename": safe_filename,
            "prediction": predicted_class,
            "confidence": float(confidence),
            "processing_time": str(datetime.now() - start_time),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/retrain", tags=["Training"])
async def retrain(files: List[UploadFile] = File(...),
                 learning_rate: float = 0.0001,
                 epochs: int = 10,
                 db: Session = Depends(get_db)):
    global model, CLASS_NAMES
    
    new_data_dir = os.path.join(UPLOAD_FOLDER, "new_data")
    os.makedirs(new_data_dir, exist_ok=True)
    
    try:
        start_time = datetime.now()
        image_paths = []
        extracted_dirs = []
        
        for file in files:
            if not allowed_file(file.filename):
                continue
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            await save_upload_file(file, file_path)
            
            if file.filename.endswith(".zip"):
                extract_dir = os.path.join(UPLOAD_FOLDER, f"extract_{os.path.splitext(file.filename)[0]}")
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                extracted_dirs.append(extract_dir)
                os.remove(file_path)
                
                for subdir in ['train', 'val']:
                    subdir_path = os.path.join(extract_dir, subdir)
                    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                        for item in os.listdir(subdir_path):
                            item_path = os.path.join(subdir_path, item)
                            if os.path.isdir(item_path) and item not in ['__MACOSX']:
                                target_dir = os.path.join(new_data_dir, item)
                                os.makedirs(target_dir, exist_ok=True)
                                for img in os.listdir(item_path):
                                    if allowed_file(img):
                                        shutil.copy(os.path.join(item_path, img), os.path.join(target_dir, img))
            else:
                image_paths.append(file_path)
        
        for img_path in image_paths:
            try:
                img_array = preprocess_image(open(img_path, 'rb').read())
                prediction = model.predict(img_array)
                label_index = np.argmax(prediction)
                label = CLASS_NAMES[label_index]
                
                label_dir = os.path.join(new_data_dir, label)
                os.makedirs(label_dir, exist_ok=True)
                img_filename = os.path.basename(img_path)
                shutil.copy(img_path, os.path.join(label_dir, img_filename))
            except Exception:
                continue
        
        # Get all valid classes from the new data directory
        class_counts = {}
        for class_dir in os.listdir(new_data_dir):
            class_path = os.path.join(new_data_dir, class_dir)
            if os.path.isdir(class_path):
                image_count = len([f for f in os.listdir(class_path) if allowed_file(f)])
                if image_count >= 2:  # Require at least 2 images per class
                    class_counts[class_dir] = image_count
                else:
                    shutil.rmtree(class_path)
        
        if not class_counts:
            raise HTTPException(status_code=400, detail="No valid classes with sufficient data found")
        
        # Use only the classes present in the new data for training
        target_names = list(class_counts.keys())
        use_validation = all(count >= 4 for count in class_counts.values())  # Require 4+ images for validation
        
        # Data generators
        if use_validation:
            data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1
            )
            train_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            validation_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )
        else:
            data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1
            )
            train_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                shuffle=True
            )
            validation_generator = None
        
        # Save and reload model to avoid issues
        temp_model_path = os.path.join(MODEL_DIR, "temp_model.keras")
        model.save(temp_model_path)
        working_model = tf.keras.models.load_model(temp_model_path)
        
        # Adjust model output layer to match the number of classes in the generator
        num_classes = len(train_generator.class_indices)
        if working_model.output_shape[-1] != num_classes:
            # Rebuild the model with the correct output layer
            base_model = tf.keras.Sequential(working_model.layers[:-1])  # Exclude the last layer
            base_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
            working_model = base_model
        
        num_layers = len(working_model.layers)
        freeze_until = int(num_layers * 0.98)
        for i, layer in enumerate(working_model.layers):
            layer.trainable = i >= freeze_until
        
        working_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if use_validation else 'loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if use_validation else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        if use_validation:
            history = working_model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                callbacks=callbacks,
                steps_per_epoch=max(1, len(train_generator)),
                validation_steps=max(1, len(validation_generator))
            )
        else:
            history = working_model.fit(
                train_generator,
                epochs=epochs,
                callbacks=callbacks,
                steps_per_epoch=max(1, len(train_generator))
            )
        
        # Evaluate the model
        if use_validation:
            validation_generator.reset()
            y_pred = working_model.predict(validation_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = validation_generator.classes
            target_names = list(validation_generator.class_indices.keys())  # Use generator classes
        else:
            train_generator.reset()
            y_pred = working_model.predict(train_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = train_generator.classes
            target_names = list(train_generator.class_indices.keys())  # Use generator classes
        
        class_report = classification_report(
            y_true,
            y_pred_classes,
            target_names=target_names,
            output_dict=True
        )
        
        save_visualizations(y_true, y_pred_classes, target_names)
        
        working_model.save(KERAS_PATH)
        model = tf.keras.models.load_model(KERAS_PATH)
        
        # Update CLASS_NAMES to include only the classes used in training
        CLASS_NAMES = list(train_generator.class_indices.keys())
        with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
            json.dump(CLASS_NAMES, f)
        
        class_metrics = {}
        for class_name in target_names:
            if class_name in class_report:
                class_metrics[class_name] = {
                    "precision": float(class_report[class_name]['precision']),
                    "recall": float(class_report[class_name]['recall']),
                    "f1_score": float(class_report[class_name]['f1-score']),
                    "support": int(class_report[class_name]['support'])
                }
        
        training_accuracy = float(history.history['accuracy'][-1]) if 'accuracy' in history.history else None
        validation_accuracy = float(history.history['val_accuracy'][-1]) if use_validation and 'val_accuracy' in history.history else None
        
        retraining = Retraining(
            num_classes=len(CLASS_NAMES),
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy,
            class_metrics=json.dumps(class_metrics)
        )
        db.add(retraining)
        db.commit()
        
        base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
        response_content = {
            "status": "success",
            "metrics": {
                "training_accuracy": training_accuracy,
                "validation_accuracy": validation_accuracy,
                "class_metrics": class_metrics
            },
            "visualization_files": {
                "classification_report": f"{base_url}/visualizations/classification_report.png",
                "confusion_matrix": f"{base_url}/visualizations/confusion_matrix.png"
            },
            "retraining_id": retraining.id,
            "processing_time": str(datetime.now() - start_time),
            "timestamp": datetime.now().isoformat()
        }
        return response_content
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
    
    finally:
        for extract_dir in extracted_dirs:
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
        if os.path.exists(new_data_dir):
            shutil.rmtree(new_data_dir)
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

@app.get("/prediction_history", tags=["History"])
async def get_prediction_history(db: Session = Depends(get_db)):
    predictions = db.query(Prediction).order_by(Prediction.timestamp.desc()).all()
    return [{"id": p.id, "text": f"Predicted disease: {p.predicted_disease}", 
             "confidence": p.confidence, "date": p.timestamp.isoformat()}
            for p in predictions]

@app.get("/retraining_history", tags=["History"])
async def get_retraining_history(db: Session = Depends(get_db)):
    retrainings = db.query(Retraining).order_by(Retraining.timestamp.desc()).all()
    return [{"id": r.id, "text": f"Retrained model with {r.num_classes} classes", 
             "training_accuracy": r.training_accuracy, "validation_accuracy": r.validation_accuracy,
             "class_metrics": json.loads(r.class_metrics) if r.class_metrics else {},
             "date": r.timestamp.isoformat()}
            for r in retrainings]

# Server Startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=300,
        reload=False
    )