import pickle
import uvicorn
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(title = "heart-failure-prediction")

"""
LOADING THE MODEL.
"""

with open('heartfailure_model.bin', 'rb') as f_in:
    dictVectorizer, model, feature_matrix = pickle.load(f_in)

class Patient(BaseModel):
    age: int = Field(..., ge = 0, le = 110, description = 'age of the patient (years)')
    sex: Literal["m", "f"]
    chestpaintype: Literal["ta", "ata", "nap", "asy"]
    restingbp: int = Field(..., ge = 0, le = 201, description = 'resting blood pressure of the patient (mm Hg)')
    cholesterol: int = Field(..., ge = 0, le = 603, description = 'serum cholesterol of the patient (mm/dl)')
    fastingbs: Literal["no", "yes"]
    restingecg: Literal["normal", "st", "lvh"]
    maxhr: int = Field(..., ge = 60, le = 202, description = 'maximum heart rate of the patient')
    exerciseangina: Literal["no", "yes"]
    oldpeak: float = Field(..., ge = -3.0, le = 7.0, description = 'old peak (measured in depression)')
    st_slope: Literal["up", "down", "flat"]


def predict_health(patient):
    X_patient = dictVectorizer.transform(patient)
    dPatient = xgb.DMatrix(X_patient, feature_names = feature_matrix)
    prediction = model.predict(dPatient)

    return float(prediction)

@app.post("/predict", response_model = dict)
def predict(patient: Patient):
    heart_failure_probability = predict_health(patient.model_dump())
    heart_failure = (heart_failure_probability > 0.5)

    return {
        "heart failure probability": heart_failure_probability,
        "heart failure": heart_failure
    }

if __name__ == "__main__":
  uvicorn.run(app, host = "localhost", port = 9696)