"""Request/response schemas for the heart-disease risk API.

Validation bounds come from the UCI Heart Disease data dictionary
(Cleveland). Categorical codes are strict enums: the model pipeline would
silently absorb unknown codes (OneHotEncoder handle_unknown="ignore"), but
for a clinical predictor an explicit 422 beats silent garbage-in  -  see
docs/DECISIONS.md.
"""
from typing import Literal

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="Sex (1 = male, 0 = female)")
    cp: Literal[1, 2, 3, 4] = Field(
        ...,
        description=(
            "Chest pain type (1=typical angina, 2=atypical angina, "
            "3=non-anginal pain, 4=asymptomatic)"
        ),
    )
    trestbps: float = Field(
        ..., ge=80, le=220, description="Resting systolic blood pressure (mm Hg)"
    )
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(
        ..., description="Fasting blood sugar > 120 mg/dl (1 = true)"
    )
    restecg: Literal[0, 1, 2] = Field(
        ...,
        description="Resting ECG (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)",
    )
    thalach: float = Field(
        ..., ge=60, le=220, description="Maximum heart rate achieved (bpm)"
    )
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(
        ...,
        ge=0.0,
        le=7.0,
        description="ST depression induced by exercise relative to rest",
    )
    slope: Literal[1, 2, 3] = Field(
        ...,
        description="Slope of peak exercise ST segment (1=up, 2=flat, 3=down)",
    )
    ca: Literal[0, 1, 2, 3] = Field(
        ..., description="Number of major vessels colored by fluoroscopy"
    )
    thal: Literal[3, 6, 7] = Field(
        ...,
        description="Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)",
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no disease, 1 = disease present")
    probability: float = Field(
        ..., description="Model probability of the disease class"
    )
    risk_label: str = Field(..., description='"disease" or "no disease"')
