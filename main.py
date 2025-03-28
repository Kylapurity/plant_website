import os
import sys
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
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
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from dotenv import load_dotenv

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
VISUALIZATION_DIR = os.path.join(current_dir, "visualizations")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_DIR = os.path.join(current_dir, "models")
KERAS_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL").replace("mysql://", "mysql+mysqlconnector://", 1)
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# JWT and password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="Plant Disease Classifier API",
             description="API for classifying plant diseases from leaf images",
             version="1.0.0")

# Mount static files for visualizations
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# ================== CORS CONFIGURATION ================== #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== DATABASE MODELS ================== #
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    predictions = relationship("Prediction", back_populates="user")
    retrainings = relationship("Retraining", back_populates="user")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    predicted_disease = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="predictions")

class Retraining(Base):
    __tablename__ = "retrainings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    num_classes = Column(Integer, nullable=False)
    training_accuracy = Column(Float)
    validation_accuracy = Column(Float, nullable=True)
    class_metrics = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="retrainings")

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
    
    model = load_model(KERAS_PATH)
    logger.info("Model loaded successfully with %d classes", len(CLASS_NAMES))
    
except Exception as e:
    logger.error("MODEL LOADING FAILED!", exc_info=True)
    raise RuntimeError(f"Model initialization failed: {str(e)}")

# ================== AUTHENTICATION HELPERS ================== #
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    from datetime import timedelta
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(db, username=username)
    if user is None:
        raise credentials_exception
    
    return user

# ================== HELPER FUNCTIONS ================== #
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

def save_visualizations(y_true, y_pred_classes, target_names):
    class_report = tf.keras.metrics.classification_report(y_true, y_pred_classes, target_names=target_names)
    plt.figure(figsize=(10, len(target_names) * 0.5 + 2))
    plt.text(0.01, 0.99, class_report, {'fontsize': 10}, fontfamily='monospace')
    plt.axis('off')
    plt.title("Classification Report")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "classification_report.png"), bbox_inches='tight', dpi=300)
    plt.close()

    cm = tf.keras.metrics.confusion_matrix(y_true, y_pred_classes)
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

# ================== API ENDPOINTS ================== #
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Plant Disease Classifier API",
        "status": "operational",
        "model_status": "loaded" if model else "not loaded",
        "class_count": len(CLASS_NAMES),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/signup", response_model=Token, tags=["Authentication"])
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, username=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict", response_model=dict, tags=["Prediction"])
async def predict(file: UploadFile = File(...), 
                 db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    try:
        start_time = datetime.now()
        
        if not file.filename or not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        await save_upload_file(file, filepath)
        
        file_size = os.path.getsize(filepath)
        if file_size > MAX_FILE_SIZE:
            os.remove(filepath)
            raise HTTPException(status_code=413, detail=f"File too large")
        
        predicted_class, confidence = predict_image(filepath, model, CLASS_NAMES)
        
        # Save to database
        prediction = Prediction(
            user_id=current_user.id,
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
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/upload", tags=["Training"])
async def upload_data(files: List[UploadFile] = File(...),
                    current_user: User = Depends(get_current_user)):
    saved_files = []
    errors = []
    start_time = datetime.now()
    
    for file in files:
        try:
            if not file.filename or not allowed_file(file.filename):
                errors.append(f"Invalid file: {file.filename}")
                continue
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
            filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
            
            await save_upload_file(file, filepath)
            
            file_size = os.path.getsize(filepath)
            if file_size > MAX_FILE_SIZE:
                os.remove(filepath)
                errors.append(f"File too large: {file.filename}")
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
async def retrain(db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    global model, CLASS_NAMES
    try:
        start_time = datetime.now()
        
        if not os.path.exists(UPLOAD_FOLDER) or not os.listdir(UPLOAD_FOLDER):
            raise HTTPException(status_code=400, detail="No training data available")
        
        metrics = retrain_model(model, UPLOAD_FOLDER)
        
        # Save visualizations
        # Note: Assuming retrain_model returns y_true, y_pred_classes, target_names
        if 'y_true' in metrics and 'y_pred_classes' in metrics and 'target_names' in metrics:
            save_visualizations(metrics['y_true'], metrics['y_pred_classes'], metrics['target_names'])
        
        # Save to database
        retraining = Retraining(
            user_id=current_user.id,
            num_classes=len(CLASS_NAMES),
            training_accuracy=metrics.get('training_accuracy'),
            validation_accuracy=metrics.get('validation_accuracy'),
            class_metrics=str(metrics.get('class_metrics', {}))
        )
        db.add(retraining)
        db.commit()
        
        # Save the updated model
        backup_path = os.path.join(MODEL_DIR, f"Model1_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
        shutil.copyfile(KERAS_PATH, backup_path)
        model.save(KERAS_PATH)
        
        base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
        return {
            "status": "success",
            "metrics": metrics,
            "backup_path": backup_path,
            "visualization_files": {
                "classification_report": f"{base_url}/visualizations/classification_report.png",
                "confusion_matrix": f"{base_url}/visualizations/confusion_matrix.png"
            },
            "retraining_id": retraining.id,
            "processing_time": str(datetime.now() - start_time),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/prediction_history", tags=["History"])
async def get_prediction_history(db: Session = Depends(get_db), 
                               current_user: User = Depends(get_current_user)):
    predictions = db.query(Prediction).filter(Prediction.user_id == current_user.id).order_by(Prediction.timestamp.desc()).all()
    return [{"id": p.id, "text": f"Predicted disease: {p.predicted_disease}", 
             "confidence": p.confidence, "date": p.timestamp.isoformat()}
            for p in predictions]

@app.get("/retraining_history", tags=["History"])
async def get_retraining_history(db: Session = Depends(get_db), 
                               current_user: User = Depends(get_current_user)):
    retrainings = db.query(Retraining).filter(Retraining.user_id == current_user.id).order_by(Retraining.timestamp.desc()).all()
    return [{"id": r.id, "text": f"Retrained model with {r.num_classes} classes", 
             "training_accuracy": r.training_accuracy, "validation_accuracy": r.validation_accuracy,
             "class_metrics": r.class_metrics, "date": r.timestamp.isoformat()}
            for r in retrainings]

@app.get("/visualize", tags=["Analytics"])
async def visualize():
    try:
        start_time = datetime.now()
        sample_data = np.random.rand(100, len(CLASS_NAMES))
        predictions = np.argmax(sample_data, axis=1)
        
        plots = {}
        plt.figure(figsize=(12, 6))
        sns.countplot(x=predictions)
        plt.xticks(ticks=range(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=90)
        plt.title('Class Distribution')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots["class_distribution"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
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
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@app.get("/health", tags=["Monitoring"])
async def health_check():
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

# ================== SERVER STARTUP ================== #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting server on port %d", port)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=300,
        reload=False
    )