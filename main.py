from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from pydantic import BaseModel
from utils import ModelInputGenerator
import numpy as np
import joblib
import zipfile
import os


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# Descomprime el modelo
zip_path = './model.pkl.zip'
# Directorio donde descomprimirás los archivos
extract_dir = './'

# Crear el directorio de extracción si no existe
os.makedirs(extract_dir, exist_ok=True)

# Descomprimir el archivo ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Definir el modelo de entrada
class MessageInput(BaseModel):
    ingredientlist: list[str]  # Lista de strings
    sugar: float

ml_models = {}
input_generator = ModelInputGenerator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["nova_model"] = joblib.load('model.pkl')
    ml_models["clf_model"] = joblib.load('clf_model.pkl')
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


def predict_risk(clf, nova_group: int, sugar_level: int):
    return clf.predict([[nova_group, sugar_level]])[0]


@app.post("/predict/")
async def predict(data: MessageInput):

    nova_model = ml_models["nova_model"]
    clf_model = ml_models["clf_model"]

    vector = input_generator.gen_input_vector(data.ingredientlist)
    print(vector)
    # 1. Obtener nova
    nova = int(np.argmax(nova_model.predict(vector))) + 1# model.predict recibe una lista
    print('result', nova)
    print(type(nova))

    # 2. Sugar level
    sugar_level = 0
    if data.sugar <= 5:
        sugar_level = 1
    elif data.sugar > 5 and data.sugar <= 25:
        sugar_level = 2
    else:
        sugar_level = 3

    print('sugar level', sugar_level)

    diabetes_impact = predict_risk(clf_model, nova, sugar_level)

    # Retornar la predicción
    return {"nova": nova, "diabetes_impact": diabetes_impact}

