import os
import sys
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
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

# ================== INITIAL SETUP ================== #
# Disable GPU and suppress TensorFlow logs
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

# ================== CONFIGURATION ================== #
UPLOAD_FOLDER = os.path.join(current_dir, "uploaded_data")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(current_dir, "models", "Model1.keras")
PICKLE_PATH = os.path.join(current_dir, "models", "Model1.pkl")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(current_dir, "models"), exist_ok=True)

app = FastAPI(title="Plant Disease Classifier API",
             description="API for classifying plant diseases from leaf images",
             version="1.0.0")

# ================== CORS CONFIGURATION ================== #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    from src.preprocessing import preprocess_image, preprocess_dataset
    from src.model import load_model, retrain_model, load_class_indices
    from src.prediction import predict_image
    
    logger.info("Loading model...")
    model = load_model(MODEL_PATH)
    class_indices = load_class_indices(PICKLE_PATH)
    
    # Fallback if class indices loading fails
    if not isinstance(class_indices, dict):
        logger.warning("Class indices not loaded properly, creating default mapping")
        class_indices = {v: k for k, v in enumerate(CLASS_NAMES)}
    
    logger.info(f"Model loaded successfully with {len(class_indices)} classes")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

# ================== HELPER FUNCTIONS ================== #
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def save_upload_file(file: UploadFile, destination: str) -> None:
    try:
        with open(destination, "wb") as buffer:
            while chunk := await file.read(8192):  # 8KB chunks
                buffer.write(chunk)
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {str(e)}")
        raise

# ================== API ENDPOINTS ================== #
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Plant Disease Classifier API",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=dict, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Classify plant disease from an uploaded image
    
    - **file**: Image file (JPG/PNG) up to 10MB
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPG, PNG are allowed"
            )
        
        # Save file with timestamp prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        await save_upload_file(file, filepath)
        
        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size > MAX_FILE_SIZE:
            os.remove(filepath)
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size is {MAX_FILE_SIZE//(1024*1024)}MB"
            )
        
        # Make prediction
        predicted_class, confidence = predict_image(filepath, model, class_indices)
        
        return {
            "filename": safe_filename,
            "prediction": predicted_class,
            "confidence": float(confidence),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.post("/upload", tags=["Training"])
async def upload_data(files: List[UploadFile] = File(...)):
    """
    Upload multiple images for retraining
    
    - **files**: List of image files (JPG/PNG)
    """
    saved_files = []
    errors = []
    
    for file in files:
        try:
            if allowed_file(file.filename):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
                filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
                
                await save_upload_file(file, filepath)
                
                if os.path.getsize(filepath) > MAX_FILE_SIZE:
                    os.remove(filepath)
                    errors.append(f"File {file.filename} too large")
                else:
                    saved_files.append(safe_filename)
        except Exception as e:
            errors.append(f"Failed to process {file.filename}: {str(e)}")
    
    return {
        "success": len(saved_files),
        "failed": len(errors),
        "saved_files": saved_files,
        "errors": errors
    }

@app.post("/retrain", tags=["Training"])
async def retrain():
    """
    Retrain model with uploaded data
    """
    try:
        logger.info("Starting model retraining...")
        metrics = retrain_model(model, UPLOAD_FOLDER)
        
        # Save the updated model
        model.save(MODEL_PATH)
        logger.info("Model retrained and saved successfully")
        
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize", tags=["Analytics"])
async def visualize():
    """
    Generate visualization plots for model analytics
    """
    try:
        # Sample data - replace with actual model statistics
        sample_data = np.random.rand(100, len(CLASS_NAMES))
        predictions = np.argmax(sample_data, axis=1)
        
        # Generate plots
        plots = {}
        
        # Class distribution plot
        plt.figure(figsize=(12, 6))
        sns.countplot(x=predictions)
        plt.xticks(ticks=range(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=90)
        plt.title('Class Distribution')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots["class_distribution"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Confidence distribution plot
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
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ================== HEALTH CHECK ================== #
@app.get("/health", tags=["Monitoring"])
async def health_check():
    """
    Service health check endpoint
    """
    try:
        # Simple model check
        test_input = np.zeros((1, 224, 224, 3))  # Adjust based on your model input shape
        model.predict(test_input)
        
        return {
            "status": "healthy",
            "model_ready": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: model not functioning properly"
        )

# ================== SERVER STARTUP ================== #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=300
    )