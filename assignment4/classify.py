"""Fixed settings, SVM sweeps and the majority-class reference."""
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from experiment import SEED, evaluate_class, pipeline

C_VALUES = (0.01, 0.1, 1, 10, 100)
GAMMAS = ("scale", 0.01, 0.1, 1, 10)


def svm(kernel, c=1, gamma="scale", scale=True):
    return pipeline(SVC(kernel=kernel, C=c, gamma=gamma, random_state=SEED), scale)


def sweep(X, y, cv, kernel, scale=True):
    gammas = GAMMAS if kernel == "rbf" else ("scale",)
    return {
        (c, gamma): evaluate_class(svm(kernel, c, gamma, scale), X, y, cv)
        for c in C_VALUES for gamma in gammas
    }


def best_setting(grid):
    return max(grid, key=lambda setting: grid[setting]["metrics"]["auc"])


def run(X, y, cv):
    return {
        "linear": sweep(X, y, cv, "linear"),
        "rbf": sweep(X, y, cv, "rbf"),
        "dummy": evaluate_class(pipeline(DummyClassifier(strategy="most_frequent")), X, y, cv),
    }

