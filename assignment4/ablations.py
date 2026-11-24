"""The four required controls, with fixed settings for feature removal."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import classify
import regress
from experiment import SEED, evaluate_class, evaluate_reg, pipeline

HOLDOUT_SEEDS = tuple(range(SEED, SEED + 10))
# Use the course baseline depth, specified before the holdout experiments.
HOLDOUT_DEPTH = 3
POLY_ALPHAS = (10, 30, 100, 300, 1000)


def run(X, num, cv, classification, regression):
    y = (num > 0).astype(int)
    reduced = X.drop(columns=["ca", "thal"])
    feature_rows = []
    for kernel in ("linear", "rbf"):
        key = classify.best_setting(classification[kernel])
        full = classification[kernel][key]
        ablated = evaluate_class(classify.svm(kernel, *key), reduced, y, cv)
        feature_rows.append((f"SVM {kernel} {key}", "auc", full, ablated))
    depth = regress.best_setting(regression["depth"])
    alpha = regress.best_setting(regression["ridge"])
    for name, model, full in [
        (f"回归树 depth={depth}", regress.tree(depth), regression["depth"][depth]),
        (f"岭回归 alpha={alpha}", regress.ridge(alpha), regression["ridge"][alpha]),
    ]:
        feature_rows.append((name, "r2", full, evaluate_reg(model, reduced, num, cv)))

    holdout = []
    holdout_diagnostics = []
    for seed in HOLDOUT_SEEDS:
        train, test = train_test_split(np.arange(len(num)), test_size=0.2, random_state=seed)
        fitted = regress.tree(HOLDOUT_DEPTH).fit(X.iloc[train], num[train])
        prediction = fitted.predict(X.iloc[test])
        error = num[test] - prediction
        squared_error = error ** 2
        holdout.append((seed, r2_score(num[test], prediction)))
        holdout_diagnostics.append({
            "seed": seed,
            "grade_counts": [int(np.sum(num[test] == grade)) for grade in range(5)],
            "target_variance": float(np.var(num[test])),
            "mse": float(np.mean(squared_error)),
            "mae": float(np.mean(np.abs(error))),
            "large_errors": int(np.sum(np.abs(error) > 1)),
            "top3_squared_share": float(np.sort(squared_error)[-3:].sum() / squared_error.sum()),
        })

    polynomial = {
        a: evaluate_reg(pipeline(
            Ridge(alpha=a), scale=True,
            extra_steps=(("quadratic", PolynomialFeatures(degree=2, include_bias=False)),
                         ("rescale", StandardScaler())),
        ), X, num, cv)
        for a in POLY_ALPHAS
    }
    return {
        "features": feature_rows,
        "raw": {k: classify.sweep(X, y, cv, k, scale=False) for k in ("linear", "rbf")},
        "holdout": holdout, "holdout_depth": HOLDOUT_DEPTH,
        "holdout_diagnostics": holdout_diagnostics, "polynomial": polynomial,
    }

