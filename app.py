# app.py
from fastapi import FastAPI
import pandas as pd
from catboost import CatBoostClassifier
from pydantic import BaseModel

app = FastAPI()

# Load the trained model
model = CatBoostClassifier()
model.load_model('model/catboost_model.cbm')

class ChurnPredictionRequest(BaseModel):
    features: dict

@app.post('/predict')
def predict_churn(request: ChurnPredictionRequest):
    features_df = pd.DataFrame([request.features])
    prediction = model.predict(features_df)
    return {'churn': bool(prediction[0])}
