# DrugPrice
The goal of this project is to show the process of building and deploying a Machine Learning Model with FastAPI and Docker.

The model is pre-trained on `drugs_train.csv`, we used `active_ingredients.csv` and `drug_label_feature_eng.csv` to build new features.
We made also other features using different hand made feature engineering techniques (tf-idf/ svd, date transformations...).

The API's main functionality is to provide a price prediction from a drug_id belonging to the `test.pkl` dataset. 
To avoid black box mode of the machine learning model, The API provides also scores explainability. We used an advanced technique based on game theory, called SHAP to explain the influence of each variable on unit predictions. 
For each provided score, we return the top 10 variables that had a positive/negative impact on the score.

To get more information about data explaratory and features engineering steps checkout the following notebooks:
* notebooks/data-analysis.ipynb
* notebooks/feature-engineering.ipynb

The retained outputs of the notebooks are:
* models/LGB3.pkl
* models/TreeExplainer_LGB3.sav

Here's a highlevel overview of the API.


Build a machine learning pipeline to train a model to predict drug prices. The output is a packaged application layout that can be deployed to production