import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "placement_model.joblib"

# Create FastAPI application
app = FastAPI(
    title="Student Placement Prediction API",
    description="Machine Learning API to predict student placement status",
    version="1.0.0"
)

# Load trained model
model = None

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)


# Student input validation
class StudentData(BaseModel):

    CGPA: float = Field(ge=0, le=10)
    Ten_Percentage: float = Field(ge=0, le=100)
    Twelve_Percentage: float = Field(ge=0, le=100)
    Internships: int = Field(ge=0, le=10)
    Projects: int = Field(ge=0, le=20)
    Aptitude_Score: int = Field(ge=0, le=100)
    Communication_Skills: int = Field(ge=1, le=10)


# Home API
@app.get("/")
def home():

    return {
        "message": "Student Placement Prediction API is running",
        "docs": "/docs"
    }


# Health Check API
@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# Prediction API
@app.post("/predict")
def predict_placement(student: StudentData):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not trained or available"
        )

    student_data = pd.DataFrame([{
        "CGPA": student.CGPA,
        "10th_Percentage": student.Ten_Percentage,
        "12th_Percentage": student.Twelve_Percentage,
        "Internships": student.Internships,
        "Projects": student.Projects,
        "Aptitude_Score": student.Aptitude_Score,
        "Communication_Skills": student.Communication_Skills
    }])

    prediction = model.predict(student_data)[0]

    status = "Placed" if prediction == 1 else "Not Placed"

    return {
        "placement_prediction": status,
        "prediction_value": int(prediction)
    }