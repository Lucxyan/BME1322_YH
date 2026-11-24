"""Regression tree depth, fold-local pruning and ridge regularization."""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from experiment import SEED, evaluate_reg, pipeline, regression_metrics

DEPTHS = (*range(1, 13), None)
ALPHAS = (0.01, 0.1, 1, 10, 30, 100, 1000)
PRUNING_QUANTILES = (0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 1.0)


def tree(depth=None, alpha=0):
    return pipeline(DecisionTreeRegressor(max_depth=depth, ccp_alpha=alpha, random_state=SEED))


def ridge(alpha):
    return pipeline(Ridge(alpha=alpha), scale=True)


def pruning_path(X, y):
    transformed = SimpleImputer(strategy="median").fit_transform(X)
    estimator = DecisionTreeRegressor(random_state=SEED)
    return np.unique(estimator.cost_complexity_pruning_path(transformed, y).ccp_alphas)


def path_alpha(path, quantile):
    return float(path[round(quantile * (len(path) - 1))])


def prune_sweep(X, y, cv):
    paths = [pruning_path(X.iloc[train], y[train]) for train, _ in cv]
    results = {}
    for quantile in PRUNING_QUANTILES:
        pred = np.empty(len(y))
        alphas = []
        models = []
        training_r2 = []
        for (train, test), path in zip(cv, paths):
            alpha = path_alpha(path, quantile)
            fitted = tree(alpha=alpha).fit(X.iloc[train], y[train])
            pred[test] = fitted.predict(X.iloc[test])
            alphas.append(alpha)
            models.append(fitted.named_steps["model"])
            training_r2.append(r2_score(y[train], fitted.predict(X.iloc[train])))
        results[quantile] = {
            "metrics": regression_metrics(y, pred, [test for _, test in cv]),
            "pred": pred, "models": models, "alphas": alphas,
            "train_r2": float(np.mean(training_r2)),
        }
    return results


def best_setting(grid, metric="r2"):
    direction = -1 if metric == "mae" else 1
    return max(grid, key=lambda key: direction * grid[key]["metrics"][metric])


def run(X, y, cv):
    return {
        "depth": {d: evaluate_reg(tree(d), X, y, cv) for d in DEPTHS},
        "prune": prune_sweep(X, y, cv),
        "ridge": {a: evaluate_reg(ridge(a), X, y, cv) for a in ALPHAS},
        "ols_scaled": pipeline(LinearRegression(), scale=True).fit(X, y),
        "lasso_check": {
            a: evaluate_reg(pipeline(Lasso(alpha=a, max_iter=10000), scale=True), X, y, cv)
            for a in (0.01, 0.1, 1, 10)
        },
        "dummy": evaluate_reg(pipeline(DummyRegressor(strategy="mean")), X, y, cv),
    }

