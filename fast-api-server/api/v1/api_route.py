from fastapi import APIRouter, HTTPException, FastAPI
import pickle
from typing import Union, Dict, List, Any, Literal, Optional
from http import HTTPStatus
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


models = {} # dict of all models

chosen_model = None
chosen_model_id = None
model_features = None # columns for model
model_params = {}

router = APIRouter()


class FitRequest(BaseModel):
    model_id: Union[str, None] = None
    model_type: Literal['naive_bayes', 'log_reg'] = 'log_reg'
    X: Dict[str, List[Union[int, float, str]]]
    y: List[int]
    hyperparams: Union[Dict[str, Any], None] = None


class PredictRequest(BaseModel):
    X: Dict[str, List[Union[int, float, str]]]


class ApiResponse(BaseModel):
    message: str
    data: Union[Dict, None] = None

class PredictResponse(BaseModel):
    y_pred: List[float]
    y_pred_proba: List[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    global chosen_model
    global model_features
    global chosen_model_id
    global model_params
    try:
        with open('models/baseline.pkl', 'rb') as f:
            log_reg_dict = pickle.load(f)

        models['log_reg'] = log_reg_dict['model']
        model_features = log_reg_dict['columns']
        chosen_model = models['log_reg']
        chosen_model_id = 'log_reg'
        model_params['log_reg'] = {
            'C' : 103,
            'max_iter' : 10000,
            'multi_class' : 'ovr',
        }

        with open('models/naive_bayes.pkl', 'rb') as f:
            naive_bayes_model = pickle.load(f)

        models['naive_bayes'] = naive_bayes_model
        model_params['naive_bayes'] = {
            'alpha' : 1e-10
        }

        yield
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))


@router.post("/fit",
             response_model=ApiResponse,
             status_code=HTTPStatus.CREATED)
async def fit(request: FitRequest):
    global chosen_model_id
    global chosen_model
    global model_features
    global model_params
    try:
        X = pd.DataFrame(request.X)
        X = X[model_features]
        y = request.y

        if len(X) == 0 or len(y) == 0:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST,
                                detail="X_train and y_train cannot be empty.")
        if len(X) != len(y):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail='X_train and y_train must ' +
                'have the same number of samples.'
            )

        classes = np.unique(y)
        model_id = request.model_id

        if model_id is not None: # train new model and set it
            if request.hyperparams is None:
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST,
                                    detail="must pass hyperperams to new model")
            else:
                hyperparams = request.hyperparams

            if request.model_type =='log_reg':
                new_model = LogisticRegression(**hyperparams)
            elif request.model_type == 'naive_bayes':
                new_model = MultinomialNB(**hyperparams)
            else:
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST,
                                    detail='Model type is not supported')

            pipeline = Pipeline([
                ('tfidf', ColumnTransformer(
                    transformers=[
                        ('tfidf',
                         TfidfVectorizer(stop_words='english'),
                         'qst_processed')
                    ],
                    remainder='passthrough'
                )),
                ('model', new_model)
            ])

            models[model_id] = pipeline
            chosen_model_id = model_id
            chosen_model = models[model_id]
            model_params[model_id] = hyperparams

        if chosen_model is None:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST, detail='No model_id provided')

        chosen_model.fit(X, y)
        y_scores = chosen_model.decision_function(X)
        y_bin = label_binarize(y, classes=classes)

        roc_curve_data = {}
        pr_curve_data = {}

        for cls in classes: # return roc-curve and pr-curve
            fpr, tpr, roc_thresholds = roc_curve(y_bin[:, cls], y_scores[:, cls])
            roc_curve_data[str(cls)] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist()
            }

            precision, recall, pr_thresholds = precision_recall_curve(
                y_bin[:, cls], y_scores[:, cls])
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
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST,
                            detail='Data not provided')

    if chosen_model is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND,
                            detail="No model set")

    model = models[chosen_model_id]
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    return ApiResponse(message=f'prediction with model {chosen_model_id} done',
                       data={'model_id': chosen_model_id,
                             'y_pred': y_pred.tolist(),
                             'y_pred_proba': y_pred_proba.tolist()}
                       )


@router.post("/set_model", response_model=ApiResponse)
async def set_model(model_id: str):
    global chosen_model
    global models
    global chosen_model_id

    if model_id not in models:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST,
                            detail='No such model exists')

    chosen_model = models[model_id]
    chosen_model_id = model_id

    return ApiResponse(message=f"Model '{model_id}' is set")


@router.get("/list_models", response_model=ApiResponse)
async def list_models():
    try:
        global model_params
        return ApiResponse(message=f"Models list",
                           data=model_params)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
