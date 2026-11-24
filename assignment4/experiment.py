"""Read the course data and evaluate estimators on the fixed folds."""
import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent


def load_data(data_dir):
    frame = pd.read_csv(data_dir / "heart_cleveland.csv")
    split = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))["cleveland"]
    indices = np.arange(len(frame))
    tests = [np.asarray(fold, dtype=int) for fold in split["folds"]]
    if (len(tests) != 5 or len(frame) != split["n"]
            or not np.array_equal(np.sort(np.concatenate(tests)), indices)):
        raise ValueError("The five course test folds must partition the data exactly once.")
    cv = [(np.setdiff1d(indices, test), test) for test in tests]
    return frame.drop(columns="num"), frame["num"].to_numpy(), cv


def pipeline(model, scale=False, extra_steps=()):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline([*steps, *extra_steps, ("model", model)])


def class_scores(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def class_metrics(y, pred, score):
    result = {"accuracy": accuracy_score(y, pred), "auc": roc_auc_score(y, score)}
    for label, suffix in [(1, ""), (0, "_neg")]:
        for name, fn in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]:
            result[name + suffix] = fn(y, pred, pos_label=label, zero_division=0)
    return result


def average(rows):
    return {key: float(mean(row[key] for row in rows)) for key in rows[0]}


def evaluate_class(model, X, y, cv):
    rows = []
    pred = np.empty(len(y), dtype=int)
    score = np.empty(len(y))
    for train, test in cv:
        fitted = clone(model).fit(X.iloc[train], y[train])
        pred[test] = fitted.predict(X.iloc[test])
        score[test] = class_scores(fitted, X.iloc[test])
        row = class_metrics(y[test], pred[test], score[test])
        row["train_auc"] = roc_auc_score(y[train], class_scores(fitted, X.iloc[train]))
        rows.append(row)
    return {"metrics": average(rows), "pred": pred, "score": score}


def regression_metrics(y, pred, tests):
    rows = [
        {"r2": r2_score(y[test], pred[test]),
         "mae": mean_absolute_error(y[test], pred[test]),
         "mse": mean_squared_error(y[test], pred[test])}
        for test in tests
    ]
    result = average(rows)
    result["rmse"] = np.sqrt(result.pop("mse"))
    return result


def evaluate_reg(model, X, y, cv):
    pred = np.empty(len(y))
    fitted_models = []
    training_r2 = []
    for train, test in cv:
        fitted = clone(model).fit(X.iloc[train], y[train])
        pred[test] = fitted.predict(X.iloc[test])
        training_r2.append(r2_score(y[train], fitted.predict(X.iloc[train])))
        fitted_models.append(fitted.named_steps["model"])
    return {"metrics": regression_metrics(y, pred, [test for _, test in cv]),
            "pred": pred, "models": fitted_models, "train_r2": float(mean(training_r2))}

