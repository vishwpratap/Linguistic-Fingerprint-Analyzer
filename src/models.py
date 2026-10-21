"""Small, documented set of scikit-learn classifiers for authorship studies."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

MODEL_LABELS = {
    "MultinomialNB": "Multinomial Naive Bayes",
    "LogisticRegression": "Logistic Regression",
    "LinearSVC": "Linear SVC",
}


def make_classifier(name: str, *, c: float = 1.0, alpha: float = 1.0, class_weight: str | None = None, random_seed: int = 42):
    """Create a classifier with explicit regularization and deterministic settings."""
    if name == "MultinomialNB":
        if alpha <= 0:
            raise ValueError("MultinomialNB alpha must be positive")
        return MultinomialNB(alpha=alpha)
    if c <= 0:
        raise ValueError("C must be positive")
    if name == "LogisticRegression":
        return LogisticRegression(C=c, max_iter=3000, class_weight=class_weight, random_state=random_seed)
    if name == "LinearSVC":
        return LinearSVC(C=c, class_weight=class_weight, max_iter=10000, random_state=random_seed)
    raise ValueError(f"Unsupported model: {name}")
