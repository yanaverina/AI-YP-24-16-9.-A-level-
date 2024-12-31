from fastapi import APIRouter, HTTPException
import pickle
from typing import Union, Dict, List, Any
from http import HTTPStatus
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report

models = {}

chosen_model = None

router = APIRouter()


class ModelConfig(BaseModel):
    hyperparameters: Dict[str, Any]
    id: str
    model_type: str

class FitRequest(BaseModel):
    X: Dict[str, List[Union[int, float, str]]]
    y: List[int]
    config: ModelConfig

class ApiResponse(BaseModel):
    message: str
    data: Union[Dict, None] = None

class PredictRequest(BaseModel):
    model_id: str
    X: Dict[str, List[Union[int, float, str]]]

class PredictResponse(BaseModel):
    model_id: str
    y_pred: List[int]
    y_pred_proba: List[float]
    metrics: Dict[str, float]


@router.on_startup
async def load_baseline():
    global models

    with open('models/baseline.pkl', 'rb') as f:
        models['baseline'] = pickle.load(f)


@router.post("/fit", response_model=ApiResponse, status_code=HTTPStatus.CREATED)
async def fit(request: FitRequest):
    try:
        X = request.X
        y = request.y

    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    X = pd.DataFrame(request.X)
    model_id = request.model_id
    if model_id not in models:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Model not found")
    model = models[model_id]
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    return PredictResponse(y_pred=y_pred, metrics=model)



