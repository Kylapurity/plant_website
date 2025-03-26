from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from dotenv import load_dotenv
import shutil
import zipfile
import logging
from prediction import load_model, preprocess_image, predict, load_class_indices
from model import retrain_model

app = FastAPI(
    title="Plant Disease API",
    description="API for plant disease classification and retraining.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

UPLOAD_FOLDER = "uploaded_data"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

MODEL_PATH = "../models/Model1.keras"
PICKLE_PATH = "../models/Model1.pkl"

# Load model and class indices at startup
try:
    model = load_model(MODEL_PATH)
    class_indices = load_class_indices(PICKLE_PATH)
    # Validate that loaded class indices match the expected class_names
    loaded_class_names = list(class_indices.keys())
    if loaded_class_names != class_names:
        logger.warning("Loaded class names from Model2.pkl do not match expected class_names. Using expected class_names.")
    logger.info("Model and class indices loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or class indices: {str(e)}")
    raise RuntimeError("Failed to load model or class indices")

@app.get("/")
def health_check():
    """Check if the API is running."""
    return {"message": "API is running"}

@app.post("/predict")
async def predict_plant_disease(file: UploadFile = File(...)):
    """
    Predict the plant disease from an uploaded image.

    Args:
        file (UploadFile): Uploaded image file (JPEG or PNG).

    Returns:
        dict: Prediction result with filename and predicted disease.
    """
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")
        
        contents = await file.read()
        image = preprocess_image(contents)
        prediction = predict(model, image, class_indices)  # Removed class_names parameter
        logger.info(f"Prediction: {prediction} for file {file.filename}")
        
        return {"filename": file.filename, "disease": prediction}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    """
    Retrain the model with a new dataset.

    Args:
        file (UploadFile): Uploaded ZIP file containing the new dataset.

    Returns:
        dict: Success message.
    """
    try:
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Please upload a ZIP file.")

        zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(zip_path, 'wb') as f:
            f.write(await file.read())

        extract_path = os.path.join(UPLOAD_FOLDER, 'dataset')
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        os.makedirs(extract_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Handle nested folder structure
        subfolders = [d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]
        if len(subfolders) == 1:
            top_level_folder = os.path.join(extract_path, subfolders[0])
            for item in os.listdir(top_level_folder):
                shutil.move(os.path.join(top_level_folder, item), extract_path)
            shutil.rmtree(top_level_folder)

        # Validate the new dataset against the expected class names
        categories = [d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]
        if not categories:
            raise HTTPException(status_code=400, detail="No valid categories found in the dataset.")

        invalid_categories = [cat for cat in categories if cat not in class_names]
        if invalid_categories:
            raise HTTPException(status_code=400, detail=f"Invalid categories found: {invalid_categories}. Expected categories: {class_names}")

        for category in categories:
            category_path = os.path.join(extract_path, category)
            if not os.listdir(category_path):
                raise HTTPException(status_code=400, detail=f"Category '{category}' is empty.")

        retrain_model(extract_path)
        global model, class_indices
        model = load_model(MODEL_PATH)
        class_indices = load_class_indices(PICKLE_PATH)
        # Validate class indices after retraining
        loaded_class_names = list(class_indices.keys())
        if loaded_class_names != class_names:
            raise HTTPException(status_code=500, detail="Class names after retraining do not match expected class_names.")
        
        return {"message": "Model retrained successfully!"}
    except Exception as e:
        logger.error(f"Error during retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
    finally:
        # Clean up uploaded files
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
# Get the port from the environment variable (Render sets this)
port = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    # Run the FastAPI app defined in api.py
    uvicorn.run("api:app", host="0.0.0.0", port=port)