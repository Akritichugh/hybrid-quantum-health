
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/api/predict",
    tags=["Disease Prediction"]
)


# Project root: hybrid-quantum-health
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# -------------------------
# Load Diabetes model
# -------------------------

DIABETES_MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "diabetes_model.pkl"
)

diabetes_model = joblib.load(DIABETES_MODEL_PATH)


# -------------------------
# Load Heart Disease model
# -------------------------

HEART_MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "heart_disease_model.joblib"
)

heart_model = joblib.load(HEART_MODEL_PATH)


# -------------------------
# Load Pneumonia model
# -------------------------

PNEUMONIA_MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "pneumonia_model.keras"
)

pneumonia_model = tf.keras.models.load_model(
    PNEUMONIA_MODEL_PATH
)


# -------------------------
# Diabetes prediction
# -------------------------

class DiabetesInput(BaseModel):
    Pregnancies: int = Field(ge=0)
    Glucose: float = Field(ge=0)
    BloodPressure: float = Field(ge=0)
    SkinThickness: float = Field(ge=0)
    Insulin: float = Field(ge=0)
    BMI: float = Field(ge=0)
    DiabetesPedigreeFunction: float = Field(ge=0)
    Age: int = Field(ge=1)


@router.post("/diabetes")
def predict_diabetes(data: DiabetesInput):

    patient_data = pd.DataFrame([data.model_dump()])

    prediction = int(
        diabetes_model.predict(patient_data)[0]
    )

    probability = float(
        diabetes_model.predict_proba(patient_data)[0][1]
    )

    return {
        "prediction": prediction,
        "estimated_probability": round(
            probability * 100, 2
        ),
        "message": (
            "Model predicted class 1"
            if prediction == 1
            else "Model predicted class 0"
        )
    }


# -------------------------
# Heart Disease prediction
# -------------------------

class HeartDiseaseInput(BaseModel):
    age: int = Field(ge=1)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: float = Field(ge=0)
    chol: float = Field(ge=0)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=0)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0)
    slope: int = Field(ge=0, le=2)
    ca: float = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=7)


@router.post("/heart")
def predict_heart_disease(data: HeartDiseaseInput):

    features = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    patient_data = pd.DataFrame(
        [data.model_dump()],
        columns=features
    )

    prediction = int(
        heart_model.predict(patient_data)[0]
    )

    probability = float(
        heart_model.predict_proba(patient_data)[0][1]
    )

    return {
        "prediction": prediction,
        "estimated_probability": round(
            probability * 100, 2
        ),
        "message": (
            "Model predicted disease present"
            if prediction == 1
            else "Model predicted no disease"
        )
    }


# -------------------------
# Pneumonia prediction
# -------------------------

@router.post("/pneumonia")
async def predict_pneumonia(
    file: UploadFile = File(...)
):

    # Accept only image files
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG chest X-ray image."
        )

    try:
        image_bytes = await file.read()

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        image = image.resize((160, 160))

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Model includes Rescaling(1/255)
        probability = float(
            pneumonia_model.predict(
                image_array,
                verbose=0
            )[0][0]
        )

        prediction = 1 if probability >= 0.5 else 0

        return {
            "prediction": prediction,
            "estimated_probability": round(
                probability * 100, 2
            ),
            "message": (
                "Model detected pneumonia-like patterns"
                if prediction == 1
                else "Model classified the image as normal"
            )
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {str(error)}"
        )
