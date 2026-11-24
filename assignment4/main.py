"""Run all analyses and regenerate the figures, tables, and optional reports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

import ablations
import classify
import regress
from experiment import ROOT, SEED, evaluate_class, evaluate_reg, load_data, pipeline


def baseline_reference(data_dir):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-B", "baselines.py", "--json"],
        cwd=data_dir, env=env, check=True, capture_output=True, encoding="utf-8",
    )
    return json.loads(result.stdout[result.stdout.index("\n{") + 1:])


def independent_baselines(X, num, cv, classification, regression):
    y = (num > 0).astype(int)
    cls = {
        "SVM (linear)": classification["linear"][1, "scale"],
        "SVM (RBF)": classification["rbf"][1, "scale"],
    }
    for name, estimator, scale in [
        ("KNN (k=5)", KNeighborsClassifier(5), True),
        ("Naive Bayes", GaussianNB(), False),
        ("Logistic Regression", LogisticRegression(max_iter=3000), True),
        ("Decision Tree (d=3)", DecisionTreeClassifier(max_depth=3, random_state=SEED), False),
    ]:
        cls[name] = evaluate_class(pipeline(estimator, scale), X, y, cv)

    reg = {"Ridge (alpha=1)": regression["ridge"][1],
           "Regression Tree (d=3)": regression["depth"][3]}
    for name, estimator, scale in [
        ("Linear Regression", LinearRegression(), False),
        ("Lasso (alpha=0.01)", Lasso(alpha=0.01), True),
        ("SVR (RBF, C=1)", SVR(C=1), True),
    ]:
        reg[name] = evaluate_reg(pipeline(estimator, scale), X, num, cv)

    return {
        f"cleveland/{task}/{name}": result["metrics"]
        for task, models in [("cls", cls), ("reg", reg)]
        for name, result in models.items()
    }


def run(data_dir):
    X, num, cv = load_data(data_dir)
    classification = classify.run(X, (num > 0).astype(int), cv)
    print("SVM sweeps complete.", flush=True)
    regression = regress.run(X, num, cv)
    print("Regression sweeps complete.", flush=True)
    reference = baseline_reference(data_dir)
    independent = independent_baselines(X, num, cv, classification, regression)
    mine = {
        model: {metric: round(independent[model][metric], 6) for metric in metrics}
        for model, metrics in reference.items()
    }
    max_difference = max(abs(mine[k][m] - v) for k, metrics in reference.items()
                         for m, v in metrics.items())
    if max_difference > 0.01:
        raise ValueError(f"Baseline verification failed. Largest absolute difference: {max_difference:.6f}")
    print("Baseline comparison passed. Running the four controls.", flush=True)
    controls = ablations.run(X, num, cv, classification, regression)
    return X, num, cv, classification, regression, controls, reference, mine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "assignment4-data")
    parser.add_argument("--write-report", action="store_true",
                        help="also regenerate report.md and analysis.md, replacing any edits")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if not all((data_dir / name).is_file() for name in ("heart_cleveland.csv", "splits.json", "baselines.py")):
        parser.error("give --data-dir pointing to the extracted course assignment4-data directory")
    from report_build import render
    render(*run(data_dir), write_report=args.write_report)
    print("Reproduced the figures and tables.", flush=True)


if __name__ == "__main__":
    main()

