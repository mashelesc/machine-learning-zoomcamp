import pickle
import uvicorn
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(title = "lung-cancer-prediction")

with open('model.bin', 'rb') as f_in:
    (model, dictVectorizer) = pickle.load(f_in)

feature_matrix = dictVectorizer.feature_names_

class Patient(BaseModel):
   gender: Literal["m", "f"]
   age: int = Field(..., ge = 0, le = 120)
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

def predict_cancer(patient_dict): 
   X_patient = dictVectorizer.transform([patient_dict])
   dPatient = xgb.DMatrix(X_patient, feature_names = feature_matrix)
   
   prediction = model.predict(dPatient)
   return float(prediction[0])

@app.post("/predict", response_model = dict)
def predict(patient: Patient):
    patient_dict = patient.model_dump()
    cancer_probability = predict_cancer(patient_dict)
    cancer = (cancer_probability > 0.5)

    return {
        "lung cancer probability": cancer_probability,
        "lung cancer": cancer
    }

if __name__ == "__main__":
  uvicorn.run(app, host = "localhost", port = 8000)
