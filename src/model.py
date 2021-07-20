from sklearn.base import RegressorMixin
import pickle
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
import time
from src.helpers import *

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

DATA_PATH = config.get('PATHS', 'DATA_PATH')
MODELS_PATH = config.get('PATHS', 'MODELS_PATH')
features_list = config.get('MODELS', 'features')

with open(os.path.join(DATA_PATH, 'metadata', features_list), 'rb') as f:
    features = pickle.load(f)

default_lgb_params = {'num_leaves': 1023,
                      'max_depth': -1,  # 15
                      'objective': 'regression',
                      'seed': 42,
                      'silent': True,
                      'num_threads': -1,
                      'n_estimators': 1000,
                      'colsample_bytree': 0.3,
                      'subsample': 0.8,
                      'learning_rate': 0.005}


class CustomLGBMRegressor(RegressorMixin):
    """
    Custom Lightgbm Regressor:
    *
    """

    def __init__(self, folds=10,
                 seed=42,
                 save_cv_rounds=True,
                 early_stopping_rounds=300,
                 eval_metric='rmse',
                 best_model_file='LGB3.pkl',
                 **lgb_params):
        """
        Initiate the light gbm regressor parameters
        :param lgb_params:
        """
        self.folds = folds
        self.seed = seed
        self.save_cv_rounds = save_cv_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.lgb_params = default_lgb_params
        self.lgb_params.update(lgb_params)
        self._clf = None
        self.best_model_file = best_model_file

    def fit(self, X=None, y=None):
        """
        Train the model with cross validation technique
        :param X: 
        :param y: 
        :return: 
        """""
        for c in X.columns:
            if X[c].dtype in ['object', 'str']:
                X[c] = X[c].astype('category')
                le = LabelEncoder()
                X[c] = le.fit_transform(X[c].astype(str))

        oof = pd.DataFrame(np.zeros(X.shape[0]), columns=["pred"])
        print(oof.shape)
        CV_SPLITTER = pd.qcut(y, self.folds, labels=False)
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        for fold, (idxT, idxV) in enumerate(skf.split(X, CV_SPLITTER)):
            print("FOLD", fold)
            X_train, y_train = X.loc[idxT, features], y[idxT]
            X_test, y_test = X.loc[idxV, features], y[idxV]

            fit_params = {"early_stopping_rounds": self.early_stopping_rounds,
                          'eval_metric': self.eval_metric,
                          "eval_set": [(X_test, y_test)],
                          'eval_names': ['valid'],
                          'feature_name': 'auto',  # that's actually the default
                          'categorical_feature': 'auto'  # that's actually the default
                          }
            clf = lgb.LGBMRegressor(**self.lgb_params)
            START = time.time()
            clf.fit(X_train, y_train, **fit_params)
            END = time.time()
            print("time to train ", round(END - START, 1), "secondes")

            if self.save_cv_rounds:
                joblib.dump(self._clf, os.path.join(f'drug_price_lgb{fold}.pkl'))
            #     clf = joblib.load(f'lgb{fold}.pkl')
            oof.loc[idxV, "pred"] = self._clf.predict(X_test[features])

        self._clf = clf
        self.cv_ = oof
        print("done")
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        if not self._clf:
            self.load_model()
        X = pandas_converter(X)
        return self._clf.predict(X)

    def load_model(self, filename=None):
        if not filename:
            filename = self.best_model_file
        self._clf = joblib.load(os.path.join(MODELS_PATH, filename))


if __name__ == "__main__":
    train = pd.read_pickle('../data/transformed/train.pkl')
    target = 'price'
    # train = pd.read_csv(f)
    y = train[target].copy()
    X = train.drop([target], 1)

    lgb = CustomLGBMRegressor(folds=3, n_estimators=10)
    #   lgb.fit(X, y)
    #   print(lgb.oof_)
    #  lgb.load_model()
    example = {'administrative_status': 1.0,
               'approved_for_hospital_use': 1.0,
               'reimbursement_rate': np.nan,
               'marketing_authorization_status': 0.0,
               'label_plaquette': 1.0,
               'label_ampoule': 0.0,
               'label_flacon': 0.0,
               'label_tube': 0.0,
               'label_stylo': 0.0,
               'label_seringue': 0.0,
               'label_pilulier': 0.0,
               'label_sachet': 0.0,
               'label_comprime': 1.0,
               'label_gelule': 0.0,
               'label_film': 0.0,
               'label_poche': 0.0,
               'label_capsule': 0.0,
               'count_plaquette': 3.0,
               'count_ampoule': 0.0,
               'count_flacon': 0.0,
               'count_tube': 0.0,
               'count_stylo': 0.0,
               'count_seringue': 0.0,
               'count_pilulier': 0.0,
               'count_sachet': 0.0,
               'count_comprime': 28.0,
               'count_gelule': 0.0,
               'count_film': 0.0,
               'count_poche': 0.0,
               'count_capsule': 0.0,
               'count_ml': 0.0,
               'declaration_year': 2012.0,
               'authorization_year': 2011.0,
               'delta_days_decralation_autorization': 365.0,
               'delta_years_decralation_autorization': 1.0,
               'dosage_form_svd_tfidf_component_0': 0.9577440997441964,
               'dosage_form_svd_tfidf_component_1': -0.26527643035193654,
               'dosage_form_svd_tfidf_component_2': 0.0024395332494299778,
               'dosage_form_svd_tfidf_component_3': -0.015241731355217737,
               'dosage_form_svd_tfidf_component_4': -1.1990452314589778e-05,
               'dosage_form_svd_tfidf_component_5': 0.00012612409750727477,
               'dosage_form_svd_tfidf_component_6': -0.002207295721455358,
               'dosage_form_svd_tfidf_component_7': -0.00048518937567741427,
               'dosage_form_svd_tfidf_component_8': -0.020191075254130244,
               'dosage_form_svd_tfidf_component_9': -0.10766794853446954,
               'dosage_form_svd_tfidf_component_10': -0.00012966082137264045,
               'dosage_form_svd_tfidf_component_11': 0.00010529923055953561,
               'dosage_form_svd_tfidf_component_12': 0.0010832587093571615,
               'dosage_form_svd_tfidf_component_13': -0.00012669452096400333,
               'dosage_form_svd_tfidf_component_14': 0.0016341825067589164,
               'dosage_form_svd_tfidf_component_15': -2.9730089953481274e-06,
               'dosage_form_svd_tfidf_component_16': 0.0012135677888257636,
               'dosage_form_svd_tfidf_component_17': -9.951652839950805e-05,
               'dosage_form_svd_tfidf_component_18': 1.2257064935401672e-12,
               'dosage_form_svd_tfidf_component_19': 0.0006305921290535857,
               'dosage_form_svd_tfidf_component_20': 5.723000688377849e-06,
               'dosage_form_svd_tfidf_component_21': 0.0004580566437673695,
               'dosage_form_svd_tfidf_component_22': -2.5470068526709584e-05,
               'dosage_form_svd_tfidf_component_23': -9.332260088386724e-07,
               'dosage_form_svd_tfidf_component_24': 4.319651804119435e-05,
               'route_of_administration_svd_tfidf_component_0': 0.999999948534149,
               'route_of_administration_svd_tfidf_component_1': -5.7749878930385916e-05,
               'route_of_administration_svd_tfidf_component_2': -2.13201370570542e-06,
               'route_of_administration_svd_tfidf_component_3': -1.006173491139246e-19,
               'route_of_administration_svd_tfidf_component_4': -2.093613980498688e-05,
               'route_of_administration_svd_tfidf_component_5': -9.552501602465667e-06,
               'route_of_administration_svd_tfidf_component_6': -0.0001089340931388823,
               'route_of_administration_svd_tfidf_component_7': -6.761839378246663e-08,
               'route_of_administration_svd_tfidf_component_8': -3.0888330365729094e-19,
               'route_of_administration_svd_tfidf_component_9': 1.3082324626196519e-08,
               'route_of_administration_svd_tfidf_component_10': -1.0625506509307516e-05,
               'route_of_administration_svd_tfidf_component_11': -6.940106423713723e-07,
               'route_of_administration_svd_tfidf_component_12': -0.00021503840141867168,
               'route_of_administration_svd_tfidf_component_13': 2.72380072184503e-05,
               'route_of_administration_svd_tfidf_component_14': 1.8634844149036447e-08,
               'route_of_administration_svd_tfidf_component_15': -3.520998328443868e-06,
               'route_of_administration_svd_tfidf_component_16': -2.4592714314511366e-08,
               'route_of_administration_svd_tfidf_component_17': -0.00011486194582310633,
               'route_of_administration_svd_tfidf_component_18': -4.754424125484155e-05,
               'route_of_administration_svd_tfidf_component_19': -1.7547471937346878e-07,
               'route_of_administration_svd_tfidf_component_20': -6.385324738020477e-07,
               'route_of_administration_svd_tfidf_component_21': -7.648730483074873e-08,
               'route_of_administration_svd_tfidf_component_22': 3.841884072511899e-05,
               'route_of_administration_svd_tfidf_component_23': 0.00011290814999786867,
               'route_of_administration_svd_tfidf_component_24': -6.356312816221881e-05,
               'country': 16.0,
               'pharmaceutical_companies_svd_tfidf_component_0': 4.054173767703e-11,
               'pharmaceutical_companies_svd_tfidf_component_1': 7.348245013100572e-06,
               'pharmaceutical_companies_svd_tfidf_component_2': 0.9998279008566043,
               'pharmaceutical_companies_svd_tfidf_component_3': -1.2057287754042352e-15,
               'pharmaceutical_companies_svd_tfidf_component_4': 4.933108081181276e-16,
               'pharmaceutical_companies_svd_tfidf_component_5': -4.893258952346152e-16,
               'pharmaceutical_companies_svd_tfidf_component_6': -5.860816703056854e-11,
               'pharmaceutical_companies_svd_tfidf_component_7': -2.270242378186057e-06,
               'pharmaceutical_companies_svd_tfidf_component_8': 8.007062662354334e-18,
               'pharmaceutical_companies_svd_tfidf_component_9': -6.913687049121031e-06,
               'pharmaceutical_companies_svd_tfidf_component_10': 4.423328843366023e-18,
               'pharmaceutical_companies_svd_tfidf_component_11': 3.0973268624747185e-19,
               'pharmaceutical_companies_svd_tfidf_component_12': 1.3436552390220159e-17,
               'pharmaceutical_companies_svd_tfidf_component_13': 1.2939260894796253e-06,
               'pharmaceutical_companies_svd_tfidf_component_14': -3.168611614832938e-18,
               'pharmaceutical_companies_svd_tfidf_component_15': 1.0396635498163451e-07,
               'pharmaceutical_companies_svd_tfidf_component_16': -9.326139048560948e-05,
               'pharmaceutical_companies_svd_tfidf_component_17': 8.802759290932451e-16,
               'pharmaceutical_companies_svd_tfidf_component_18': 8.936217213089606e-16,
               'pharmaceutical_companies_svd_tfidf_component_19': 3.43387731029366e-08,
               'pharmaceutical_companies_svd_tfidf_component_20': 3.008818028095611e-07,
               'pharmaceutical_companies_svd_tfidf_component_21': 1.485881040397999e-08,
               'pharmaceutical_companies_svd_tfidf_component_22': 1.933076651547793e-06,
               'pharmaceutical_companies_svd_tfidf_component_23': -2.358771892345339e-10,
               'pharmaceutical_companies_svd_tfidf_component_24': -4.308752868323065e-08}
    print(lgb.predict(example))
