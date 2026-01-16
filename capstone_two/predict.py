import pickle
import uvicorn
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(title = "lung-cancer-prediction")

output_file = f'model.bin'

with open(output_file, 'rb') as f_in:
    dictVectorizer, model, feature_matrix = pickle.load(f_in)

class Patient(BaseModel):
   gender: Literal["no", "yes"]
   age: int = Field(..., ge = 0, le = 130)
   smoking: Literal["no", "yes"]
   yellow_fingers: Literal["no", "yes"]
   anxiety: Literal["no", "yes"]
   peer_pressure: Literal["no", "yes"]
   chronic_disease: Literal["no", "yes"]
   fatigue: Literal["no", "yes"]
   allergy: Literal["no", "yes"]
   wheezing: Literal["no", "yes"]
   alcohol_consuming: Literal["no", "yes"]
   coughing: Literal["no", "yes"]
   shortness_of_breath: Literal["no", "yes"]
   swallowing_difficulty: Literal["no", "yes"]
   chest_pain: Literal["no", "yes"]

def predict_cancer(patient):
   X_patient = dictVectorizer.transform(patient)
   dPatient = xgb.DMatrix(X_patient, feature_names = feature_matrix)
   
   prediction = model.predict(dPatient)
   return float(prediction)

@app.post("/predict", response_model = dict)
def predict(patient: Patient):
    lung_cancer_probability = predict_cancer(patient.model_dump())
    lung_cancer = (lung_cancer_probability > 0.5)

    return {
        "lung cancer probability": lung_cancer_probability,
        "lung cancer": lung_cancer
    }

if __name__ == "__main__":
  uvicorn.run(app, host = "localhost", port = 9696)