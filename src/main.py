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

# Add the parent directory of 'src' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_image, preprocess_dataset
from src.model import load_model, retrain_model, load_class_indices
from src.prediction import predict_image

app = FastAPI()

# Enable CORS to allow the React frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = "uploaded_data"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = "../models/Model1.keras"
PICKLE_PATH = "../models/Model1.pkl"

# Define class names explicitly
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

# Load models globally
model = load_model(MODEL_PATH)
class_indices = load_class_indices(PICKLE_PATH)

# Debug: Verify the types
print(f"Type of model: {type(model)}")
print(f"Type of class_indices: {type(class_indices)}")

# Validate that class_indices match class_names
if not isinstance(class_indices, dict):
    raise ValueError(f"Expected class_indices to be a dict, got {type(class_indices)}")
if list(class_indices.keys()) != class_names:
    raise ValueError("Class names in pickle file do not match defined class_names")

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/")
async def root():
    return {"message": "Welcome to the Plant Disease Classifier API"}

# Model Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    # Save the uploaded file
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    # Predict
    predicted_class, confidence = predict_image(filepath)
    return {"prediction": predicted_class, "confidence": confidence}

# Upload Data for Retraining
@app.post("/upload")
async def upload_data(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    for file in files:
        if not allowed_file(file.filename):
            continue
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
    
    return {"message": f"{len(files)} files uploaded successfully"}

# Trigger Retraining with Metrics
@app.post("/retrain")
async def retrain():
    try:
        metrics = retrain_model(model, UPLOAD_FOLDER)
        model.save(MODEL_PATH)
        return {"message": "Model retrained successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Visualizations
@app.get("/visualize")
async def visualize():
    # Placeholder data (replace with actual dataset analysis)
    sample_data = np.random.rand(100, len(class_names))
    predictions = np.argmax(sample_data, axis=1)
    
    # Feature 1: Class Distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(x=predictions)
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=90)
    plt.title('Distribution of Predicted Classes')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    class_dist_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Feature 2: Confidence Scores
    confidence_scores = np.max(sample_data, axis=1)
    plt.figure(figsize=(10, 6))
    sns.histplot(confidence_scores, bins=20)
    plt.title('Distribution of Confidence Scores')
    plt.xlabel('Confidence')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    confidence_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Feature 3: Healthy vs Diseased
    healthy_count = np.sum([1 for p in predictions if 'healthy' in class_names[p]])
    diseased_count = len(predictions) - healthy_count
    plt.figure(figsize=(6, 6))
    plt.pie([healthy_count, diseased_count], labels=['Healthy', 'Diseased'], autopct='%1.1f%%')
    plt.title('Healthy vs Diseased Plants')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    health_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return {
        "class_distribution": class_dist_img,
        "confidence_scores": confidence_img,
        "health_status": health_img
    }

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use Render's PORT or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)