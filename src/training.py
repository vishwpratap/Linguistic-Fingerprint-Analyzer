"""Leakage-aware model selection and held-out authorship evaluation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from .dataset_manager import DatasetValidationError, train_validation_test_split
from .feature_pipeline import REPRESENTATIONS, make_feature_transformer
from .models import MODEL_LABELS, make_classifier


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[Any]) -> dict[str, Any]:
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(p)), "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f)), "weighted_f1": float(precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)[2]),
        "per_author": pd.DataFrame({"author": labels, "precision": p, "recall": r, "f1": f, "support": support}),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels,
    }


def run_experiments(
    frame: pd.DataFrame,
    *,
    representations: list[str] | None = None,
    models: list[str] | None = None,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_seed: int = 42,
    group_column: str | None = None,
    minimum_samples_per_author: int = 5,
    max_vocabulary: int = 3000,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    include_pos: bool = False,
    class_weight: str | None = None,
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0),
    nb_alphas: tuple[float, ...] = (0.1, 1.0),
    style_weight: float = 1.0,
    text_weight: float = 1.0,
    word_case_sensitive: bool = False,
    char_case_sensitive: bool = False,
    remove_stop_words: bool = False,
) -> dict[str, Any]:
    """Fit all candidates on train, select by validation macro-F1, test once.

    The returned candidate table contains validation scores only. The held-out
    test set is used once, after winner selection, and is reported separately.
    """
    representations = list(representations or ["stylometric", "word_unigram_bigram", "character", "hybrid_word"])
    models = list(models or list(MODEL_LABELS))
    unknown = [name for name in representations if name not in REPRESENTATIONS]
    if unknown:
        raise ValueError(f"Unknown representations: {unknown}")
    unknown_models = [name for name in models if name not in MODEL_LABELS]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    if not representations or not models:
        raise ValueError("Select at least one feature representation and one model.")
    if not c_values or any(value <= 0 for value in c_values) or not nb_alphas or any(value <= 0 for value in nb_alphas):
        raise ValueError("C and alpha candidate values must be positive.")

    split = train_validation_test_split(
        frame, test_size=test_size, validation_size=validation_size,
        random_seed=random_seed, group_column=group_column,
        minimum_samples_per_author=minimum_samples_per_author,
    )
    train = frame.iloc[split["train_indices"]]
    validation = frame.iloc[split["validation_indices"]]
    test = frame.iloc[split["test_indices"]]
    labels = sorted(frame["author"].unique().tolist(), key=str)
    candidates = []
    fitted = {}
    for representation in representations:
        for model_name in models:
            values = [("alpha", value) for value in nb_alphas] if model_name == "MultinomialNB" else [("C", value) for value in c_values]
            for parameter, value in values:
                feature_transform = make_feature_transformer(
                    representation, max_vocabulary=max_vocabulary,
                    word_ngram_range=word_ngram_range, char_ngram_range=char_ngram_range,
                    include_pos=include_pos, style_weight=style_weight, text_weight=text_weight,
                    word_case_sensitive=word_case_sensitive, char_case_sensitive=char_case_sensitive,
                    remove_stop_words=remove_stop_words,
                )
                classifier = make_classifier(
                    model_name, c=value if parameter == "C" else 1.0,
                    alpha=value if parameter == "alpha" else 1.0,
                    class_weight=class_weight, random_seed=random_seed,
                )
                pipeline = Pipeline([("features", feature_transform), ("classifier", classifier)])
                started = time.perf_counter()
                pipeline.fit(train["text"].astype(str).tolist(), train["author"].to_numpy())
                fit_seconds = time.perf_counter() - started
                predictions = pipeline.predict(validation["text"].astype(str).tolist())
                metrics = _metrics(validation["author"].to_numpy(), predictions, labels)
                candidate_id = f"{representation}|{model_name}|{parameter}={value:g}"
                candidates.append({
                    "candidate_id": candidate_id,
                    "representation": representation,
                    "representation_label": REPRESENTATIONS[representation],
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "parameter": f"{parameter}={value:g}",
                    "validation_accuracy": metrics["accuracy"],
                    "validation_macro_precision": metrics["macro_precision"],
                    "validation_macro_recall": metrics["macro_recall"],
                    "validation_macro_f1": metrics["macro_f1"],
                    "validation_weighted_f1": metrics["weighted_f1"],
                    "fit_seconds": fit_seconds,
                })
                fitted[candidate_id] = (pipeline, fit_seconds)
    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["validation_macro_f1", "validation_accuracy", "candidate_id"], ascending=[False, False, True]
    ).reset_index(drop=True)
    winner_id = str(candidate_frame.iloc[0]["candidate_id"])
    winner_pipeline, winner_fit_seconds = fitted[winner_id]
    test_texts = test["text"].astype(str).tolist()
    started = time.perf_counter()
    test_predictions = winner_pipeline.predict(test_texts)
    prediction_seconds = time.perf_counter() - started
    test_metrics = _metrics(test["author"].to_numpy(), test_predictions, labels)
    misclassified = []
    correctly_classified = []
    for (_, row), predicted in zip(test.iterrows(), test_predictions):
        raw_text = str(row["text"])
        example = {"actual": row["author"], "predicted": predicted, "text": raw_text[:450] + ("…" if len(raw_text) > 450 else "")}
        if row["author"] != predicted:
            misclassified.append(example)
        else:
            correctly_classified.append(example)
    confusion = test_metrics["confusion_matrix"]
    confusion_pairs = [
        {"actual": labels[row], "predicted": labels[column], "count": int(confusion[row, column])}
        for row in range(len(labels)) for column in range(len(labels))
        if row != column and confusion[row, column] > 0
    ]
    confusion_pairs.sort(key=lambda item: (-item["count"], str(item["actual"]), str(item["predicted"])))

    # Descriptive author profiles are calculated from training data only.
    profile_rows = []
    from .feature_pipeline import StylometricFeatureExtractor
    raw_style = StylometricFeatureExtractor(include_pos=False).fit_transform(train["text"].astype(str).tolist())
    style_names = StylometricFeatureExtractor(include_pos=False).get_feature_names_out()
    for author in labels:
        mask = train["author"].to_numpy() == author
        if mask.any():
            means = np.mean(raw_style[mask], axis=0)
            for name, value in zip(style_names, means):
                profile_rows.append({"author": author, "feature": name, "train_mean": float(value)})
    return {
        "winner_id": winner_id,
        "winner_pipeline": winner_pipeline,
        "winner": candidate_frame.iloc[0].to_dict(),
        "candidates": candidate_frame,
        "test_metrics": test_metrics,
        "test_fit_seconds": winner_fit_seconds,
        "test_prediction_ms_per_sample": prediction_seconds * 1000 / max(len(test_texts), 1),
        "test_misclassifications": misclassified,
        "test_correct_classifications": correctly_classified,
        "common_confusions": confusion_pairs,
        "split": split,
        "authors": labels,
        "author_profiles": pd.DataFrame(profile_rows),
        "test_truth": test["author"].to_numpy(),
        "test_predictions": test_predictions,
        "train_rows": train.copy(),
        "validation_rows": validation.copy(),
        "test_rows": test.copy(),
        "config": {
            "representations": representations, "models": models, "max_vocabulary": max_vocabulary,
            "word_ngram_range": word_ngram_range, "char_ngram_range": char_ngram_range,
            "include_pos": include_pos, "class_weight": class_weight,
            "minimum_samples_per_author": minimum_samples_per_author,
            "c_values": c_values, "nb_alphas": nb_alphas,
            "style_weight": style_weight, "text_weight": text_weight,
            "word_case_sensitive": word_case_sensitive, "char_case_sensitive": char_case_sensitive,
            "remove_stop_words": remove_stop_words,
            "selection_metric": "validation macro-F1",
        },
    }
