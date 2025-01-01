from fastapi import APIRouter, HTTPException, Depends, FastAPI
import pickle
from typing import Union, Dict, List, Any
from http import HTTPStatus
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import label_binarize

models = {}

chosen_model = None
chosen_model_id = None
model_features = None

router = APIRouter()


class ModelConfig(BaseModel):
    hyperparameters: Dict[str, Any]
    id: str
    model_type: str


class FitRequest(BaseModel):
    X: Dict[str, List[Union[int, float, str]]]
    y: List[int]
    config: ModelConfig


class PredictRequest(BaseModel):
    model_id: str
    X: Dict[str, List[Union[int, float, str]]]


class ApiResponse(BaseModel):
    message: str
    data: Union[Dict, None] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    global chosen_model
    global model_features
    try:
        with open('models/baseline.pkl', 'rb') as f:
            log_reg_dict = pickle.load(f)

        models['log_reg'] = log_reg_dict['model']
        model_features = log_reg_dict['columns']
        chosen_model = models['log_reg']

        with open('models/naive_bayes.pkl', 'rb') as f:
            naive_bayes_model = pickle.load(f)

        models['naive_bayes'] = naive_bayes_model
        yield
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))



@router.post("/fit", response_model=ApiResponse, status_code=HTTPStatus.CREATED)
async def fit(request: FitRequest):
    global chosen_model_id
    global chosen_model
    global model_features

    try:
        X = pd.DataFrame(request.X)
        X = X[model_features]
        y = request.y

        if len(X) == 0 or len(y) == 0:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST,
                                detail="X_train and y_train cannot be empty.")
        if len(X) != len(y):
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST,
                                detail="X_train and y_train must have the same number of samples.")

        classes = np.unique(y)

        if chosen_model is None:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail='No model set')

        chosen_model.fit(X, y)
        y_scores = chosen_model.decision_function(X)
        y_bin = label_binarize(y, classes=classes)

        roc_curve_data = {}
        pr_curve_data = {}

        for i, cls in enumerate(classes):
            fpr, tpr, roc_thresholds = roc_curve(y_bin[:, i], y_scores[:, i])
            roc_curve_data[str(cls)] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist()
            }

            precision, recall, pr_thresholds = precision_recall_curve(y_bin[:, i], y_scores[:, i])
            pr_curve_data[str(cls)] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist()
            }

        return ApiResponse(message=f'Model {chosen_model_id} trained',
                           data={
                               "roc_curve": roc_curve_data,
                               "pr_curve": pr_curve_data}
                           )

    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post("/predict", response_model=ApiResponse)
async def predict(request: PredictRequest):
    global chosen_model_id

    try:
        X = pd.DataFrame(request.X)
        X = X[model_features]
    except Exception:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail='Data not provided')

    if chosen_model is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="No model set")

    model = models[chosen_model_id]
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    return ApiResponse(message=f'prediction with model {chosen_model_id} done',
                       data={'model_id': chosen_model_id,
                             'y_pred': y_pred,
                             'y_pred_proba': y_pred_proba}
                       )


@router.post("/set_model", response_model=ApiResponse)
async def set_model(model_id: str):
    global chosen_model
    global models

    if model_id not in models:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail='No such model exists')

    chosen_model = models[model_id]
    chosen_model_id = model_id

    return ApiResponse(message=f"Model '{model_id}' is set")


@router.get("/list_models", response_model=ApiResponse)
async def list_models():
    try:
        global models
        models_params = {}
        for model_id in models.keys():
            params = models[model_id].get_params()
            params = {key: str(value) for key, value in params.items()}
            models_params[model_id] = params

        return ApiResponse(message=f"Models list",
                           data={'models_params': [models_params]})
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
