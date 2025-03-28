import os
import sys
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import List
import tensorflow as tf
import uvicorn
from datetime import datetime
import logging
import shutil

# Setup: Disable GPU and suppress TensorFlow logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fix path resolution for Render
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

# Configuration
UPLOAD_FOLDER = os.path.join(current_dir, "uploaded_data")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_DIR = os.path.join(current_dir, "models")
KERAS_PATH = os.path.join(MODEL_DIR, "Model.keras")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Log directory structure for debugging
logger.info("Current directory: %s", current_dir)
logger.info("Directory contents:")
for root, dirs, files in os.walk(current_dir):
    logger.info("%s has files: %s and dirs: %s", root, files, dirs)

app = FastAPI(title="Plant Disease Classifier API",
              description="API for classifying plant diseases from leaf images",
              version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class names
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

# Model loading
try:
    from src.preprocessing import preprocess_image, preprocess_dataset
    from src.model import load_model, retrain_model
    from src.prediction import predict_image
    
    logger.info("Attempting to load model from: %s", KERAS_PATH)
    
    if not os.path.exists(KERAS_PATH):
        available_files = "\n".join(os.listdir(MODEL_DIR)) if os.path.exists(MODEL_DIR) else "Directory does not exist"
        raise FileNotFoundError(
            f"Model file not found at {KERAS_PATH}\n"
            f"Available files in models directory:\n{available_files}"
        )
    
    logger.info("Model file found, proceeding with loading...")
    logger.info("Loading Keras model...")
    model = load_model(KERAS_PATH)
    
    logger.info("Model loaded successfully with %d classes", len(CLASS_NAMES))
    
    logger.info("Running test prediction to verify model...")
    test_input = np.zeros((1, 128, 128, 3))
    prediction = model.predict(test_input)
    logger.info("Test prediction successful, output shape: %s", prediction.shape)
    
except Exception as e:
    logger.error("MODEL LOADING FAILED!", exc_info=True)
    logger.error("Current working directory: %s", os.getcwd())
    logger.error("Directory contents: %s", os.listdir('.'))
    if os.path.exists('src'):
        logger.error("src directory contents: %s", os.listdir('src'))
    if os.path.exists('models'):
        logger.error("models directory contents: %s", os.listdir('models'))
    
    raise RuntimeError(
        f"Model initialization failed: {str(e)}\n"
        f"Current directory: {os.getcwd()}\n"
        f"Looking for model at: {KERAS_PATH}"
    )

# Helper functions
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def save_upload_file(file: UploadFile, destination: str) -> None:
    try:
        with open(destination, "wb") as buffer:
            while chunk := await file.read(8192):
                buffer.write(chunk)
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {str(e)}")
        if os.path.exists(destination):
            os.remove(destination)
        raise

# API endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Plant Disease Classifier API",
        "status": "operational",
        "model_status": "loaded" if model else "not loaded",
        "class_count": len(CLASS_NAMES),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=dict, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """Classify plant disease from an image
    
    - **file**: Image file (JPG/PNG, max 10MB)
    """
    try:
        start_time = datetime.now()
        
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPG, PNG are allowed")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        await save_upload_file(file, filepath)
        
        file_size = os.path.getsize(filepath)
        if file_size > MAX_FILE_SIZE:
            os.remove(filepath)
            raise HTTPException(status_code=413, detail=f"File too large. Max size is {MAX_FILE_SIZE//(1024*1024)}MB")
        
        logger.info("Starting prediction for file: %s", safe_filename)
        predicted_class, confidence = predict_image(filepath, model, CLASS_NAMES)
        logger.info("Prediction completed in %s", datetime.now() - start_time)
        
        return {
            "filename": safe_filename,
            "prediction": predicted_class,
            "confidence": float(confidence),
            "processing_time": str(datetime.now() - start_time),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")

@app.post("/upload", tags=["Training"])
async def upload_data(files: List[UploadFile] = File(...)):
    """Upload images for retraining
    
    - **files**: List of image files (JPG/PNG)
    """
    saved_files = []
    errors = []
    start_time = datetime.now()
    
    for file in files:
        try:
            if not file.filename:
                errors.append("Empty filename")
                continue
                
            if not allowed_file(file.filename):
                errors.append(f"Invalid file type: {file.filename}")
                continue
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
            filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
            
            await save_upload_file(file, filepath)
            
            file_size = os.path.getsize(filepath)
            if file_size > MAX_FILE_SIZE:
                os.remove(filepath)
                errors.append(f"File {file.filename} too large ({file_size//1024}KB)")
            else:
                saved_files.append(safe_filename)
                
        except Exception as e:
            errors.append(f"Failed to process {file.filename}: {str(e)}")
    
    return {
        "success": len(saved_files),
        "failed": len(errors),
        "saved_files": saved_files,
        "errors": errors,
        "processing_time": str(datetime.now() - start_time)
    }

@app.post("/retrain", tags=["Training"])
async def retrain(epochs: int = 5, batch_size: int = 32):
    """Retrain the model with uploaded images
    
    Returns metrics: accuracy, precision, recall, F1 score, and success status.
    
    - **epochs**: Number of training epochs (default: 5)
    - **batch_size**: Batch size for training (default: 32)
    """
    try:
        start_time = datetime.now()
        logger.info("Starting model retraining with epochs=%d, batch_size=%d", epochs, batch_size)
        
        if not os.path.exists(UPLOAD_FOLDER) or not os.listdir(UPLOAD_FOLDER):
            raise HTTPException(status_code=400, detail="No training data available. Upload images first.")
        
        # Call retrain_model and get metrics
        result = retrain_model(model, UPLOAD_FOLDER, epochs=epochs, batch_size=batch_size)
        
        # Check if retraining was successful
        if not result.get('success', False):
            raise HTTPException(status_code=500, detail=f"Retraining failed: {result.get('error', 'Unknown error')}")
        
        # Save the updated model
        logger.info("Saving retrained model...")
        backup_path = os.path.join(MODEL_DIR, f"Model1_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
        shutil.copyfile(KERAS_PATH, backup_path)
        model.save(KERAS_PATH)
        
        logger.info("Model retrained and saved successfully")
        
        # Prepare response with metrics
        response = {
            "success": True,
            "metrics": {
                "val_loss": result['val_loss'],
                "accuracy": result['val_accuracy'],
                "precision": result['precision'],
                "recall": result['recall'],
                "f1_score": result['f1_score']
            },
            "backup_path": backup_path,
            "processing_time": str(datetime.now() - start_time),
            "timestamp": datetime.now().isoformat()
        }
        return response
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "processing_time": str(datetime.now() - start_time),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/visualize", tags=["Analytics"])
async def visualize(
    num_samples: int = Query(100, description="Number of samples to generate for visualization"),
    plot_type: str = Query("both", description="Type of plot: 'class_distribution', 'confidence', or 'both'")
):
    """Generate plots for model analytics
    
    - **num_samples**: Number of samples to generate (default: 100)
    - **plot_type**: Type of plot to generate (default: 'both')
    """
    try:
        start_time = datetime.now()
        
        sample_data = np.random.rand(num_samples, len(CLASS_NAMES))
        predictions = np.argmax(sample_data, axis=1)
        
        plots = {}
        
        if plot_type in ["class_distribution", "both"]:
            plt.figure(figsize=(12, 6))
            sns.countplot(x=predictions)
            plt.xticks(ticks=range(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=90)
            plt.title('Class Distribution')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plots["class_distribution"] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
        
        if plot_type in ["confidence", "both"]:
            plt.figure(figsize=(10, 6))
            sns.histplot(np.max(sample_data, axis=1), bins=20)
            plt.title('Confidence Distribution')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plots["confidence"] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
        
        return {
            "status": "success",
            "plots": plots,
            "processing_time": str(datetime.now() - start_time),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization generation failed: {str(e)}")

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Service health check"""
    try:
        start_time = datetime.now()
        
        test_input = np.zeros((1, 128, 128, 3))
        prediction = model.predict(test_input)
        
        upload_dir_ok = os.path.exists(UPLOAD_FOLDER)
        
        return {
            "status": "healthy",
            "model_ready": True,
            "model_output_shape": str(prediction.shape),
            "upload_dir_accessible": upload_dir_ok,
            "response_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.get("/debug/files", tags=["Debug"])
async def debug_files():
    """Check file structure"""
    def list_files(path):
        if not os.path.exists(path):
            return f"Path does not exist: {path}"
        return {"path": path, "files": os.listdir(path)}
    
    return {
        "current_dir": list_files('.'),
        "src_dir": list_files('src'),
        "models_dir": list_files('models'),
        "upload_dir": list_files(UPLOAD_FOLDER),
        "abs_paths": {
            "keras": KERAS_PATH,
            "exists": {"keras": os.path.exists(KERAS_PATH)}
        }
    }

# Server startup
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting server on port %d", port)
    logger.info("Current working directory: %s", os.getcwd())
    logger.info("Environment variables: %s", {k: v for k, v in os.environ.items() if 'PYTHON' in k or 'PORT' in k})
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=300,
        reload=False
    )