"""Robustness checks for length, source leakage, ablation, and preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .dataset_manager import DatasetValidationError, train_validation_test_split
from .preprocessing import tokenize_words
from .training import run_experiments


def _score_subset(truth: np.ndarray, prediction: np.ndarray, labels: list[Any]) -> dict[str, Any]:
    if len(truth) == 0:
        return {"samples": 0, "accuracy": None, "macro_f1": None, "per_author": []}
    precision, recall, f1, support = precision_recall_fscore_support(truth, prediction, labels=labels, zero_division=0)
    return {
        "samples": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "per_author": pd.DataFrame({"author": labels, "precision": precision, "recall": recall, "f1": f1, "support": support}),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels),
    }


def evaluate_document_length_robustness(result: dict[str, Any], *, thresholds: tuple[int, ...] = (50, 100, 200), minimum_group_samples: int = 10) -> dict[str, Any]:
    """Score the fixed winner on held-out test subsets across minimum word lengths."""
    test = result["test_rows"].copy()
    truth = np.asarray(result["test_truth"])
    prediction = np.asarray(result["test_predictions"])
    lengths = test["text"].astype(str).map(lambda text: len(tokenize_words(text))).to_numpy()
    labels = result["authors"]
    rows = []
    for threshold in sorted(set(int(value) for value in thresholds)):
        if threshold < 0:
            raise ValueError("Length thresholds must be nonnegative word counts")
        selected = lengths >= threshold
        metric = _score_subset(truth[selected], prediction[selected], labels)
        rows.append({
            "minimum_words": threshold, "samples": metric["samples"],
            "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"],
            "warning": "Too few held-out samples for a stable interpretation." if metric["samples"] < minimum_group_samples else None,
            "per_author": metric["per_author"],
        })
    bins = [("short (<50 words)", 0, 50), ("medium (50–99 words)", 50, 100), ("long (100+ words)", 100, None)]
    group_rows = []
    for name, lower, upper in bins:
        selected = lengths >= lower if upper is None else (lengths >= lower) & (lengths < upper)
        metric = _score_subset(truth[selected], prediction[selected], labels)
        group_rows.append({
            "length_group": name, "samples": metric["samples"], "accuracy": metric["accuracy"],
            "macro_f1": metric["macro_f1"],
            "warning": "Too few held-out samples for a stable interpretation." if metric["samples"] < minimum_group_samples else None,
            "per_author": metric["per_author"],
        })
    return {"thresholds": pd.DataFrame([{key: value for key, value in row.items() if key != "per_author"} for row in rows]),
            "threshold_details": rows, "length_groups": pd.DataFrame([{key: value for key, value in row.items() if key != "per_author"} for row in group_rows]),
            "length_group_details": group_rows}


def evaluate_domain_robustness(result: dict[str, Any], *, column: str, minimum_group_samples: int = 10) -> dict[str, Any]:
    """Report held-out metrics by a supplied genre/topic domain label."""
    test = result["test_rows"].reset_index(drop=True)
    if column not in test.columns:
        raise DatasetValidationError(f"Domain column '{column}' is not present in the held-out test rows.")
    truth = np.asarray(result["test_truth"])
    prediction = np.asarray(result["test_predictions"])
    labels = result["authors"]
    values = test[column].where(test[column].notna(), "[missing]").astype(str)
    rows, details = [], []
    for value in sorted(values.unique()):
        selected = values.eq(value).to_numpy()
        metric = _score_subset(truth[selected], prediction[selected], labels)
        row = {"domain": value, "samples": metric["samples"], "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"],
               "warning": "Too few held-out samples for a stable domain comparison." if metric["samples"] < minimum_group_samples else None}
        rows.append(row)
        details.append({**row, "per_author": metric["per_author"]})
    return {"column": column, "summary": pd.DataFrame(rows), "details": details,
            "warning": "Domain differences can reflect class balance, sample size, topic, source, or other distribution shifts; this table does not determine the cause."}


def compare_random_and_group_splits(frame: pd.DataFrame, *, group_column: str, run_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Run identical model-selection protocols under random and grouped splits."""
    output = {}
    rows = []
    for name, selected_group in (("stratified passage split", None), (f"grouped by {group_column}", group_column)):
        try:
            result = run_experiments(frame, **{**run_kwargs, "group_column": selected_group})
            output[name] = result
            metric = result["test_metrics"]
            rows.append({"split": name, "method": result["split"]["method"], "winner": result["winner_id"],
                         "test_samples": result["split"]["sizes"]["test"], "test_macro_f1": metric["macro_f1"], "status": "available"})
        except (DatasetValidationError, ValueError) as error:
            rows.append({"split": name, "method": selected_group or "stratified", "winner": None,
                         "test_samples": 0, "test_macro_f1": None, "status": str(error)})
    summary = pd.DataFrame(rows)
    available = summary.dropna(subset=["test_macro_f1"])
    delta = None
    if len(available) == 2:
        delta = float(available.iloc[1]["test_macro_f1"] - available.iloc[0]["test_macro_f1"])
    return {"summary": summary, "results": output, "grouped_minus_random_macro_f1": delta,
            "interpretation": "A decrease under grouped evaluation can indicate source leakage or distribution shift; this comparison alone does not identify the cause."}


def run_feature_ablation(frame: pd.DataFrame, *, run_kwargs: dict[str, Any], model: str) -> dict[str, Any]:
    """Compare traditional, word, character, and hybrid features on one protocol."""
    reps = ["stylometric", "word_unigram_bigram", "character", "hybrid_word"]
    results = {}
    for representation in reps:
        results[representation] = run_experiments(frame, **{**run_kwargs, "representations": [representation], "models": [model]})
    baseline = results["stylometric"]["test_metrics"]["macro_f1"]
    rows = []
    per_author_rows = []
    for representation, result in results.items():
        metrics = result["test_metrics"]
        rows.append({"representation": representation, "winner": result["winner_id"],
                     "test_macro_f1": metrics["macro_f1"], "difference_from_stylometric": metrics["macro_f1"] - baseline})
        for row in metrics["per_author"].to_dict("records"):
            per_author_rows.append({"representation": representation, **row})
    return {"summary": pd.DataFrame(rows), "per_author": pd.DataFrame(per_author_rows), "results": results,
            "warning": "Ablation comparisons are descriptive; small datasets may make per-author differences unstable."}


def run_preprocessing_sensitivity(frame: pd.DataFrame, *, run_kwargs: dict[str, Any], model: str) -> dict[str, Any]:
    """Compare text normalization variants using the same split seed/protocol."""
    configurations = [
        ("word unigrams · lowercase", "word_unigram", False, False, False, (1, 1), (3, 5)),
        ("word unigrams · original case", "word_unigram", True, False, False, (1, 1), (3, 5)),
        ("word uni+bigram · lowercase", "word_unigram_bigram", False, False, False, (1, 2), (3, 5)),
        ("word uni+bigram · remove stop words", "word_unigram_bigram", False, False, True, (1, 2), (3, 5)),
        ("character 3–5 · lowercase", "character", False, False, False, (1, 2), (3, 5)),
        ("character 2–6 · preserve case", "character", False, True, False, (1, 2), (2, 6)),
    ]
    results = {}
    rows = []
    for label, representation, word_case, char_case, remove_stops, word_range, char_range in configurations:
        result = run_experiments(frame, **{
            **run_kwargs, "representations": [representation], "models": [model],
            "word_case_sensitive": word_case, "char_case_sensitive": char_case,
            "remove_stop_words": remove_stops, "word_ngram_range": word_range,
            "char_ngram_range": char_range,
        })
        results[label] = result
        rows.append({"configuration": label, "representation": representation,
                     "test_macro_f1": result["test_metrics"]["macro_f1"],
                     "test_accuracy": result["test_metrics"]["accuracy"],
                     "test_samples": result["split"]["sizes"]["test"]})
    return {"summary": pd.DataFrame(rows), "results": results,
            "warning": "Sensitivity differences reflect the corpus, fixed split, and chosen model; they do not establish a general preprocessing effect."}


def error_analysis(result: dict[str, Any]) -> dict[str, Any]:
    """Return held-out examples, per-author scores, and common confusions."""
    return {
        "correct_examples": result["test_correct_classifications"],
        "misclassified_examples": result["test_misclassifications"],
        "per_author": result["test_metrics"]["per_author"],
        "confusion_matrix": result["test_metrics"]["confusion_matrix"],
        "labels": result["test_metrics"]["labels"],
        "common_confusions": pd.DataFrame(result["common_confusions"], columns=["actual", "predicted", "count"]),
    }
