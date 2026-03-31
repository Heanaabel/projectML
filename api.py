from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI(title="ML Models API")

#chargement les modèles sauvegardés
CLASS_MODEL_PATH = "best_model_Random_Forest.pkl"  # classification Breast Cancer
REG_MODEL_PATH = "best_model_Regression.pkl"      # regression California

if not os.path.exists(CLASS_MODEL_PATH) or not os.path.exists(REG_MODEL_PATH):
    raise FileNotFoundError("Les modèles sauvegardés n'ont pas été trouvés ! "
                            "Exécute main.py pour entraîner et sauvegarder les modèles.")

classification_model = joblib.load(CLASS_MODEL_PATH)
regression_model = joblib.load(REG_MODEL_PATH)


#définition des schemas d'entrée
class ClassInput(BaseModel):
    features: list

class RegInput(BaseModel):
    features: list


#Endpoint Classification
@app.post("/predict_class")
def predict_class(data: ClassInput):
    X = np.array(data.features).reshape(1, -1)
    prediction = classification_model.predict(X)
    return {"prediction": int(prediction[0])}


#Endpoint Regression
@app.post("/predict_reg")
def predict_reg(data: RegInput):
    X = np.array(data.features).reshape(1, -1)
    prediction = regression_model.predict(X)
    return {"prediction": float(prediction[0])}


#Endpoint test
@app.get("/")
def read_root():
    return {"message": "API ML Models fonctionne !"}