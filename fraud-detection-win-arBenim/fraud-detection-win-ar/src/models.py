import os, joblib, numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from .utils import ensure_dirs

def train_isolation_forest(X_train, cfg, models_dir):
    ensure_dirs(models_dir)
    p = cfg['isolation_forest']
    model = IsolationForest(
        n_estimators=p['n_estimators'],
        contamination=p['contamination'],
        max_features=p['max_features'],
        max_samples=p['max_samples'],
        random_state=p['random_state'],
    )
    model.fit(X_train)
    path = os.path.join(models_dir, 'isolation_forest.pkl')
    joblib.dump(model, path)
    return model, path

def evaluate_isolation_forest(model, X_test):
    y_pred = model.predict(X_test)  # 1 normal, -1 anomaly
    y_pred_mapped = (y_pred == -1).astype(int)
    return y_pred_mapped, None

def train_xgboost_smote(X_train, y_train, cfg, models_dir):
    ensure_dirs(models_dir)
    p = cfg['xgboost']
    X_res, y_res = X_train, y_train
    if p.get('use_smote', True):
        sm = SMOTE(sampling_strategy=p.get('smote_strategy','minority'), random_state=p['random_state'])
        X_res, y_res = sm.fit_resample(X_train, y_train)
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=p['n_estimators'],
        learning_rate=p['learning_rate'],
        eval_metric=p['eval_metric'],
        random_state=p['random_state'],
        use_label_encoder=False,
    )
    model.fit(X_res, y_res)
    path = os.path.join(models_dir, 'xgboost_smote.pkl')
    joblib.dump(model, path)
    return model, path

def predict_proba_xgb(model, X):
    return model.predict_proba(X)[:,1]

def evaluate_xgboost_with_threshold(model, X_test, proba_threshold=0.5):
    proba = predict_proba_xgb(model, X_test)
    y_pred = (proba >= proba_threshold).astype(int)
    return y_pred, proba

def load_model(path):
    return joblib.load(path)
