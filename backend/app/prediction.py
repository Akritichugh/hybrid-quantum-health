
from pathlib import Path
from io import BytesIO
import shap
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import json

from PIL import Image
from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    File,
    Depends,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel, Field

from app.auth import get_db
from app.models import User, PredictionHistory
from app.security import decode_access_token


router = APIRouter(
    prefix="/api/predict",
    tags=["Disease Prediction"]
)

security = HTTPBearer()


# =========================================
# AUTHENTICATION
# =========================================

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    token = credentials.credentials

    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token."
            )

        user_id = int(user_id)

    except (InvalidTokenError, ValueError, TypeError):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token."
        )

    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="User associated with this token was not found."
        )

    return user


# =========================================
# SAVE PREDICTION HISTORY
# =========================================

def save_prediction(
    db: Session,
    user: User,
    disease: str,
    prediction: int,
    probability: float,
    message: str,
):
    history = PredictionHistory(
        user_id=user.id,
        disease=disease,
        prediction=prediction,
        estimated_probability=probability,
        message=message,
    )

    db.add(history)
    db.commit()

    return {
        "prediction": prediction,
        "estimated_probability": probability,
        "message": message,
    }


# =========================================
# MODEL PATHS
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================
# LOAD DIABETES MODEL
# =========================================

DIABETES_MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "diabetes_model.pkl"
)

diabetes_model = joblib.load(DIABETES_MODEL_PATH)

# =========================================
# SHAP BACKGROUND DATA
# =========================================

DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Representative reference values for the initial SHAP explanation.
# Replace these with actual training samples when available.
diabetes_background = pd.DataFrame(
    [
        [0, 85, 60, 15, 0, 22.0, 0.10, 21],
        [1, 100, 65, 20, 30, 25.0, 0.20, 25],
        [2, 110, 70, 25, 60, 28.0, 0.30, 30],
        [3, 120, 72, 28, 80, 30.0, 0.40, 35],
        [4, 130, 75, 30, 100, 32.0, 0.50, 40],
        [5, 140, 80, 32, 120, 35.0, 0.70, 45],
        [6, 150, 82, 35, 150, 38.0, 1.00, 50],
        [8, 160, 85, 40, 180, 42.0, 1.50, 60],
    ],
    columns=DIABETES_FEATURES,
)


# =========================================
# LOAD HEART DISEASE MODEL
# =========================================

HEART_MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "heart_disease_model.joblib"
)

heart_model = joblib.load(HEART_MODEL_PATH)

# =========================================
# HEART DISEASE SHAP BACKGROUND DATA
# =========================================

HEART_FEATURES = [
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

# Representative reference values for SHAP.
# These are example reference samples, not real patients.

heart_background = pd.DataFrame(
    [
        [45, 1, 0, 120, 200, 0, 0, 160, 0, 0.0, 1, 0, 2],
        [50, 0, 1, 130, 220, 0, 1, 150, 0, 1.0, 1, 0, 2],
        [55, 1, 2, 140, 240, 0, 0, 145, 1, 1.5, 1, 1, 3],
        [60, 1, 0, 150, 260, 1, 1, 130, 1, 2.0, 2, 1, 3],
        [65, 0, 2, 135, 230, 0, 0, 140, 0, 0.5, 1, 0, 2],
        [40, 1, 3, 125, 190, 0, 1, 170, 0, 0.0, 0, 0, 2],
        [58, 1, 1, 145, 250, 0, 0, 135, 1, 1.8, 2, 2, 3],
        [62, 0, 0, 130, 210, 0, 1, 155, 0, 0.3, 1, 0, 2],
    ],
    columns=HEART_FEATURES,
)


# =========================================
# LOAD PNEUMONIA MODEL
# =========================================

PNEUMONIA_MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "pneumonia_model.keras"
)

pneumonia_model = tf.keras.models.load_model(
    PNEUMONIA_MODEL_PATH
)

PNEUMONIA_IMAGE_SIZE = (160, 160)


# =========================================
# DIABETES INPUT
# =========================================

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
def predict_diabetes(
    data: DiabetesInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient_data = pd.DataFrame([data.model_dump()])

    prediction = int(
        diabetes_model.predict(patient_data)[0]
    )

    probability = float(
        diabetes_model.predict_proba(patient_data)[0][1]
    )

    probability = round(probability * 100, 2)

    message = (
        "Model predicted diabetes risk"
        if prediction == 1
        else "Model predicted lower diabetes risk"
    )

    return save_prediction(
        db=db,
        user=current_user,
        disease="Diabetes",
        prediction=prediction,
        probability=probability,
        message=message,
    )

# =========================================
# DIABETES SHAP EXPLAINABILITY
# =========================================

@router.post("/diabetes/explain")
def explain_diabetes(
    data: DiabetesInput,
    current_user: User = Depends(get_current_user),
):
    patient_data = pd.DataFrame(
        [data.model_dump()],
        columns=DIABETES_FEATURES,
    )

    try:
        # If the saved model is a scikit-learn Pipeline, separate
        # preprocessing from its final estimator before using SHAP.
        if hasattr(diabetes_model, "steps") and len(diabetes_model.steps) > 1:
            preprocessing = diabetes_model[:-1]
            classifier = diabetes_model.steps[-1][1]

            background_transformed = preprocessing.transform(
                diabetes_background
            )
            patient_transformed = preprocessing.transform(
                patient_data
            )
        elif hasattr(diabetes_model, "steps"):
            classifier = diabetes_model.steps[-1][1]
            background_transformed = diabetes_background
            patient_transformed = patient_data
        else:
            classifier = diabetes_model
            background_transformed = diabetes_background
            patient_transformed = patient_data

        background_array = np.asarray(background_transformed)
        patient_array = np.asarray(patient_transformed)

        # Use TreeExplainer when supported; otherwise use KernelExplainer.
        try:
            explainer = shap.TreeExplainer(
                classifier,
                data=background_array,
            )
            shap_values = explainer.shap_values(patient_array)
            base_values = explainer.expected_value
        except Exception:
            explainer = shap.KernelExplainer(
                classifier.predict_proba,
                background_array,
            )
            shap_values = explainer.shap_values(
                patient_array,
                nsamples=100,
            )
            base_values = explainer.expected_value

        # Extract SHAP contributions for the positive class (class 1).
        if isinstance(shap_values, list):
            class_values = np.asarray(shap_values[1])
        else:
            class_values = np.asarray(shap_values)
            if class_values.ndim == 3 and class_values.shape[-1] > 1:
                class_values = class_values[:, :, 1]

        contributions = class_values[0].flatten()

        base_array = np.asarray(base_values)
        if base_array.size > 1:
            base_value = float(base_array.flatten()[1])
        else:
            base_value = float(base_array.flatten()[0])

        # Keep original feature names when transformed dimensions match.
        if len(contributions) == len(DIABETES_FEATURES):
            feature_names = DIABETES_FEATURES
        else:
            feature_names = [
                f"Transformed feature {index + 1}"
                for index in range(len(contributions))
            ]

        feature_explanations = []
        for feature, contribution in zip(
            feature_names,
            contributions,
        ):
            feature_explanations.append(
                {
                    "feature": feature,
                    "contribution": round(float(contribution), 6),
                }
            )

        feature_explanations.sort(
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )

        return {
            "disease": "Diabetes",
            "base_value": round(base_value, 6),
            "features": feature_explanations,
            "note": (
                "Educational model explanation only. Contributions are "
                "relative to representative reference data and are not "
                "a medical diagnosis."
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not generate SHAP explanation: "
                f"{str(error)}"
            ),
        )


# =========================================
# HEART DISEASE INPUT
# =========================================

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
def predict_heart_disease(
    data: HeartDiseaseInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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

    probability = round(probability * 100, 2)

    message = (
        "Model predicted disease risk"
        if prediction == 1
        else "Model predicted lower disease risk"
    )

    return save_prediction(
        db=db,
        user=current_user,
        disease="Heart Disease",
        prediction=prediction,
        probability=probability,
        message=message,
    )

# =========================================
# HEART DISEASE SHAP EXPLAINABILITY
# =========================================

@router.post("/heart/explain")
def explain_heart_disease(
    data: HeartDiseaseInput,
    current_user: User = Depends(get_current_user),
):
    patient_data = pd.DataFrame(
        [data.model_dump()],
        columns=HEART_FEATURES,
    )

    try:
        # Separate preprocessing and classifier if the model
        # is a scikit-learn Pipeline.
        if hasattr(heart_model, "steps") and len(heart_model.steps) > 1:
            preprocessing = heart_model[:-1]
            classifier = heart_model.steps[-1][1]

            background_transformed = preprocessing.transform(
                heart_background
            )

            patient_transformed = preprocessing.transform(
                patient_data
            )

        elif hasattr(heart_model, "steps"):
            classifier = heart_model.steps[-1][1]

            background_transformed = heart_background
            patient_transformed = patient_data

        else:
            classifier = heart_model

            background_transformed = heart_background
            patient_transformed = patient_data

        background_array = np.asarray(
            background_transformed
        )

        patient_array = np.asarray(
            patient_transformed
        )

        # Try TreeExplainer first.
        # Use KernelExplainer if the model is not supported.
        try:
            explainer = shap.TreeExplainer(
                classifier,
                data=background_array,
            )

            shap_values = explainer.shap_values(
                patient_array
            )

            base_values = explainer.expected_value

        except Exception:
            explainer = shap.KernelExplainer(
                classifier.predict_proba,
                background_array,
            )

            shap_values = explainer.shap_values(
                patient_array,
                nsamples=100,
            )

            base_values = explainer.expected_value

        # Extract SHAP values for class 1.
        if isinstance(shap_values, list):
            class_values = np.asarray(
                shap_values[1]
            )

        else:
            class_values = np.asarray(
                shap_values
            )

            if (
                class_values.ndim == 3
                and class_values.shape[-1] > 1
            ):
                class_values = class_values[:, :, 1]

        contributions = class_values[0].flatten()

        # Extract base value for class 1.
        base_array = np.asarray(
            base_values
        )

        if base_array.size > 1:
            base_value = float(
                base_array.flatten()[1]
            )
        else:
            base_value = float(
                base_array.flatten()[0]
            )

        # Use original feature names when dimensions match.
        if len(contributions) == len(HEART_FEATURES):
            feature_names = HEART_FEATURES

        else:
            feature_names = [
                f"Transformed feature {index + 1}"
                for index in range(len(contributions))
            ]

        feature_explanations = []

        for feature, contribution in zip(
            feature_names,
            contributions,
        ):
            feature_explanations.append(
                {
                    "feature": feature,
                    "contribution": round(
                        float(contribution),
                        6,
                    ),
                }
            )

        # Sort by strongest absolute contribution.
        feature_explanations.sort(
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )

        return {
            "disease": "Heart Disease",
            "base_value": round(
                base_value,
                6,
            ),
            "features": feature_explanations,
            "note": (
                "Educational model explanation only. Contributions "
                "are relative to representative reference data and "
                "are not a medical diagnosis."
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not generate Heart Disease SHAP explanation: "
                f"{str(error)}"
            ),
        )

# =========================================
# PNEUMONIA X-RAY PREDICTION
# =========================================

@router.post("/pneumonia")
async def predict_pneumonia(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/pjpeg",
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

        image = image.resize(PNEUMONIA_IMAGE_SIZE)

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        model_output = pneumonia_model.predict(
            image_array,
            verbose=0
        )

        pneumonia_probability = float(
            model_output[0][0]
        )

        prediction = (
            1 if pneumonia_probability >= 0.5 else 0
        )

        probability = round(
            pneumonia_probability * 100,
            2
        )

        message = (
            "Model detected pneumonia-like patterns"
            if prediction == 1
            else "Model predicted a normal X-ray pattern"
        )

        return save_prediction(
            db=db,
            user=current_user,
            disease="Pneumonia",
            prediction=prediction,
            probability=probability,
            message=message,
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the X-ray image: {str(error)}"
        )

# =========================================
# GET USER PREDICTION HISTORY
# =========================================

@router.get("/history")
def get_prediction_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    history = (
        db.query(PredictionHistory)
        .filter(PredictionHistory.user_id == current_user.id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )

    return [
        {
            "id": item.id,
            "disease": item.disease,
            "prediction": item.prediction,
            "estimated_probability": item.estimated_probability,
            "message": item.message,
            "created_at": item.created_at,
        }
        for item in history
    ]

# =========================================
# MODEL PERFORMANCE METRICS
# =========================================

@router.get("/performance")
def get_model_performance(
    current_user: User = Depends(get_current_user),
):
    metrics_path = (
        PROJECT_ROOT
        / "ml"
        / "models"
        / "performance_metrics.json"
    )

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Performance metrics file not found. "
                   "Please run the model evaluation script first.",
        )

    try:
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        # Load pneumonia comparison metrics
        pneumonia_comparison_path = (
        PROJECT_ROOT
            / "ml"
            / "models"
            / "pneumonia_comparison_metrics.json"
        )

        if pneumonia_comparison_path.exists():
            with open(
                pneumonia_comparison_path,
                "r",
                encoding="utf-8"
            ) as file:
                pneumonia_comparison = json.load(file)

            metrics.update(pneumonia_comparison)

        return metrics

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load performance metrics: {str(error)}",
        )
