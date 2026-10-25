"""From-scratch Bernoulli Naive Bayes for the assignment's raw MNIST protocol.

Only NumPy is used. Input pixels must be on the original 0..255 scale.
Each pixel is ink when its value is strictly greater than 128.
"""

import numpy as np


class NaiveBayesClassifier:
    """Ten-class Bernoulli model with additive smoothing and log inference."""

    THRESHOLD = 128
    N_FEATURES = 784
    N_CLASSES = 10
    BATCH_SIZE = 2048

    def __init__(self, alpha: float = 1.0):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite, positive number")
        self.alpha = float(alpha)

    @classmethod
    def _validate_X(cls, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != cls.N_FEATURES:
            raise ValueError("X must have shape (n_samples, 784)")
        if X.dtype.kind not in "biuf":
            raise ValueError("X must contain numeric pixels")
        if not np.isfinite(X).all() or np.any(X < 0) or np.any(X > 255):
            raise ValueError("X must contain finite pixel values in [0, 255]")
        return X

    def fit(self, X, y):
        """Estimate priors and pixel likelihoods; return self.

        P(Y=c) = (N_c + alpha) / (N + 10*alpha)
        P(B_j=1 | Y=c) = (N_cj + alpha) / (N_c + 2*alpha)
        """
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.shape != (len(X),) or not len(X):
            raise ValueError("y must have shape (n_samples,) with n_samples > 0")
        if y.dtype.kind not in "biuf" or not np.isfinite(y).all():
            raise ValueError("y must contain integer labels from 0 to 9")
        if np.any(y != np.floor(y)) or np.any(y < 0) or np.any(y > 9):
            raise ValueError("y must contain integer labels from 0 to 9")
        y = y.astype(np.int64)
        self.classes_ = np.arange(self.N_CLASSES)
        self.class_count_ = np.bincount(y, minlength=self.N_CLASSES)
        self.feature_count_ = np.zeros((self.N_CLASSES, self.N_FEATURES))
        # Counts use only the supplied training set. No test data or labels.
        for c in self.classes_:
            self.feature_count_[c] = np.count_nonzero(
                X[y == c] > self.THRESHOLD, axis=0
            )
        self.class_prior_ = (self.class_count_ + self.alpha) / (
            len(y) + self.N_CLASSES * self.alpha
        )
        self.feature_prob_ = (self.feature_count_ + self.alpha) / (
            self.class_count_[:, None] + 2 * self.alpha
        )
        self.class_log_prior_ = np.log(self.class_prior_)
        self.feature_log_prob_ = np.log(self.feature_prob_)
        self.feature_log_neg_prob_ = np.log1p(-self.feature_prob_)
        # B log(p) + (1-B) log(1-p) = B log(p/(1-p)) + log(1-p).
        self._log_odds = self.feature_log_prob_ - self.feature_log_neg_prob_
        self._offset = self.class_log_prior_ + self.feature_log_neg_prob_.sum(axis=1)
        return self

    def _joint_log_likelihood(self, X):
        if not hasattr(self, "_log_odds"):
            raise RuntimeError("Call fit before prediction")
        X = self._validate_X(X)
        scores = np.empty((len(X), self.N_CLASSES), dtype=np.float64)
        for start in range(0, len(X), self.BATCH_SIZE):
            stop = start + self.BATCH_SIZE
            binary = (X[start:stop] > self.THRESHOLD).astype(np.float64)
            scores[start:stop] = binary @ self._log_odds.T + self._offset
        return scores

    def predict(self, X):
        """Return shape-(n_samples,) integer labels, including empty batches."""
        scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Return normalized probabilities; column c corresponds to digit c.

        Subtracting each row's maximum prevents exponentiation overflow.
        These model posteriors need not be calibrated confidence estimates.
        """
        scores = self._joint_log_likelihood(X)
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        return weights / weights.sum(axis=1, keepdims=True)
