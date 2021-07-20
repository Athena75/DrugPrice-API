# DrugPrice
The goal of this project is to show the process of building and deploying a Machine Learning Model with FastAPI and Docker.

The model is pre-trained on `drugs_train.csv`, we used `active_ingredients.csv` and `drug_label_feature_eng.csv` to build new features.
We made also other features using different hand made feature engineering techniques (tf-idf/ svd, date transformations...).

The API's main functionality is to provide a price prediction from a drug_id belonging to the `test.pkl` dataset. 
To avoid black box mode of the machine learning model, The API provides also scores explainability. We used an advanced technique based on game theory, called SHAP to explain the influence of each variable on unit predictions. 
For each provided score, we return the top 10 variables that had a positive/negative impact on the score.

To get more information about data explaratory and features engineering steps checkout the following notebooks:
* `notebooks/data-analysis.ipynb` : the Data explaratory analysis step : allowing to get interesting insights and ideas of features engineering to apply
* `notebooks/feature-engineering.ipynb` : the retained outputs are:
    * **data/transformed/train.pkl** : transformed training dataset: this dataset is passed to the model to be trained on
    * **data/transformed/test.pkl**: transformed test dataset: it will be used by the API to get single data inputs from a test drug id, that will be passed to the model to get score prediction
* `notebooks/modeling.ipynb`: in this notebook you have all the modeling part (the lightGBM validated with stratified crosss validation technique, and the TreExplorer used to explain the model)
  The retained outputs of the notebooks are:
  * **models/LGB3.pkl** : a pre-trained lightgbm with a Cross-Validation method
  * **models/TreeExplainer_LGB3.sav** : a TreeExplainer pre-trained on the generated model: it will ba applied directeley by the api to extract the top features 

Here's a highlevel overview of the API:

![alt text](https://github.com/Athena75/DrugPrice-API/blob/main/deliveries/docs/api-overview.png?raw=true)

It contains the following routes:

* HTTP POST : `/single_score` : the body request is a single key/value dict (the drug_id to predict): 
the response is composed of :
    * the input drug_id
    * the related predicted score
    * the top shapley values explaining the score
  
![alt text](https://github.com/Athena75/DrugPrice-API/blob/main/deliveries/docs/single_score.png?raw=true)

* HTTP POST : `/bulk_score`: the body request is composed of a list of drug_ids we want to predict, the response is composed of a list of the single_score response, each item is related to a drug_id of the list input 

![alt text](https://github.com/Athena75/DrugPrice-API/blob/main/deliveries/docs/bulk_score.png?raw=true)



## Develop the app locally
The API route is defined in api/main.py

A Dockerfile is defined to create an image for this API: pretty standard.

A docker-compose to manage services between others (as for now, there's only one service: the API)

```
$ git clone https://github.com/Athena75/DrugPrice-API.git
$ cd DrugPrice-API
$ docker-compose up --build
```

To check the API is doc, you can open http://localhost:8000/docs: it would redirect you to the interactive interface where you can try the API from the browser. You get something like this:

![alt text](https://github.com/Athena75/DrugPrice-API/blob/main/deliveries/docs/docs.png?raw=true)

The index page contains the list of the **drug_id** extracted from `test.csv`, that you can test with our implemented routes

Other ways to try the API request:
* Postman 
* Curl cmd : 
  
example:
```
curl --location --request POST 'http://localhost:8000/bulk_score' \
--header 'Content-Type: application/json' \
--data-raw '{
    "drug_ids": [
        "2_test",
        "10_test",
        "15_test"
    ]
}'
```

![alt text](https://github.com/Athena75/DrugPrice-API/blob/main/deliveries/docs/bulk_curl.png?raw=true)


