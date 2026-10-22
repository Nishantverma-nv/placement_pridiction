import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from src.config import DEFAULT_MODEL, METRICS, MODELS_DIR, PRE_TEST, PRE_TRAIN


def split(data):
    X = data.drop(columns=["PlacementStatus"])
    y = data["PlacementStatus"]
    return X, y


def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=40, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=14, min_samples_leaf=10,
            n_jobs=-1, random_state=42,
        ),
        # histogram-based gradient boosting (the LightGBM/XGBoost idea, sklearn build)
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=8,
            random_state=42,
        ),
    }


def evaluate(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba[:, 1]), 4),
        "clipped": round((y_proba[:, 1] > 0.5).mean(), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run_training():
    train, test = pd.read_csv(PRE_TRAIN), pd.read_csv(PRE_TEST)
    Xtr, ytr = split(train)
    Xte, yte = split(test)

    metrics, models = {}, {}
    for name, model in make_models().items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)

        metric = evaluate(yte, pred, proba)
        if isinstance(model, LogisticRegression):
            metric["coefs"] = dict(zip(Xtr.columns, np.round(model.coef_[0], 4).tolist()))
        if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
            metric["importances"] = dict(zip(Xtr.columns, np.round(model.feature_importances_, 4).tolist()))

        metrics[name] = metric
        models[name] = model
        key = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{key}.pkl")
        print(f"[trained] {name}: accuracy={metric['accuracy']} roc_auc={metric['roc_auc']}")

    best = max(models, key=lambda n: metrics[n]["f1"])
    joblib.dump(models[best], DEFAULT_MODEL)

    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics, models, models[best]


def best_model_name(metrics):
    return max(metrics, key=lambda n: metrics[n]["f1"])


if __name__ == "__main__":
    run_training()