from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to ["http://localhost:3000"] for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once during server startup
model = tf.keras.models.load_model('model.keras')

# List of disease classes
class_name = [
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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_bytes = await file.read()
    
    # Convert the byte data to a numpy array
    image = Image.open(BytesIO(image_bytes))
    img = np.array(image.convert('RGB'))  # Convert to RGB format

    # Resize the image to match the model input size
    img = cv2.resize(img, (128, 128))

    # Preprocess the image as required by the model
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])  # Convert single image to batch

    # Make prediction using the model
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)  # Get the index of the class with the highest probability

    # Get the predicted disease name
    model_prediction = class_name[result_index]

    # Return the prediction in a JSON response
    return JSONResponse(content={"prediction": model_prediction})