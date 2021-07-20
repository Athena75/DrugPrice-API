from src.pipeline import pipe, get_shap_explain
from src.custom_transformer import LGBTransformer
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from enum import Enum
import pandas as pd
import pickle
import configparser
from zipfile import ZipFile



config = configparser.ConfigParser()
config.read('config.ini')

DATA_PATH = config.get('PATHS', 'DATA_PATH')
MODELS_PATH = config.get('PATHS', 'MODELS_PATH')
explainer_model = config.get('MODELS', 'explainer_model')

with ZipFile(os.path.join(MODELS_PATH, explainer_model), 'r') as zipObj:
   # Extract all the contents of zip file in MODELS_PATH directory
   zipObj.extractall(MODELS_PATH)

load_explainer = pickle.load(open(os.path.join(MODELS_PATH, explainer_model.strip('.zip')), 'rb'))
data = pd.read_pickle(os.path.join(DATA_PATH, 'transformed', 'test.pkl'))
data = data.set_index('drug_id')
lgb_transformer = LGBTransformer()

with open(os.path.join(DATA_PATH, 'metadata', 'train_features.list'), 'rb') as f:
    features = pickle.load(f)


class SingleRequestId(BaseModel):
    drug_id: str


class MultipleRequestId(BaseModel):
    drug_ids: List[str]


class ScoreExplain(BaseModel):
    drug_id: str
    score: float
    explain: dict


class ScoresExplain(BaseModel):
    bulk: List[ScoreExplain]


app = FastAPI()


# Defining path operation for root endpoint
@app.get('/')
def index():
    return {'dugs_ids': list(data.index)}


@app.get("/drugs_id/{drug_id}")
async def get_data_from_drug_id(drug_id: str):
    return {"drug_id": drug_id,
            "features": list(data.loc[drug_id, :].to_dict())}


@app.post("/single_score", response_model=ScoreExplain)
def single_score_with_explain(user_request: SingleRequestId):
    drug_id = user_request.drug_id
    X = data.loc[drug_id, :]
    score = pipe.predict(X)
    X = lgb_transformer.transform(X)
    explain = get_shap_explain(X, features, load_explainer, top=10)
    return ScoreExplain(drug_id=drug_id, score=score[0], explain=explain)


@app.post("/bulk_score", response_model=ScoresExplain)
def bulk_score_with_explain(user_request: MultipleRequestId):
    drug_ids = user_request.drug_ids
    X = data.loc[drug_ids, :]
    preds = pipe.predict(X)
    preds = pd.Series(preds, index=X.index)
    result = []
    X = lgb_transformer.transform(X)
    for index, row in X.iterrows():
        explain = get_shap_explain(row, features, load_explainer, top=10)
        print("INDEX", index)
        print(preds[index])
        print(explain)
        new_element = ScoreExplain(drug_id=index, score=preds[index], explain=explain)
        result.append(new_element)

    return {"bulk": result}
