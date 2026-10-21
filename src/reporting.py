"""Portable Markdown experiment reports."""

from __future__ import annotations

from typing import Any

import pandas as pd


def experiment_report(
    result: dict[str, Any], *, provenance: str = "Not provided",
    license_note: str = "Not provided", include_error_text: bool = False,
    robustness: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Format dataset provenance, split design, validation selection, and test results."""
    winner = result["winner"]
    metrics = result["test_metrics"]
    lines = [
        "# Linguistic Fingerprint Analyzer — authorship experiment",
        "",
        "> Exploratory text classification. A model prediction is not proof of identity, authorship, or legal attribution.",
        "",
        "## Dataset and provenance",
        "",
        f"- Provenance: {provenance or 'Not provided'}",
        f"- Licence / permission note: {license_note or 'Not provided'}",
        f"- Usable rows: {result['split']['sizes']['train'] + result['split']['sizes']['validation'] + result['split']['sizes']['test']}",
        f"- Authors: {', '.join(map(str, result['authors']))}",
        f"- Split method: {result['split']['method']}",
        f"- Random seed: {result['split']['random_seed']}",
        f"- Split class counts: {result['split']['class_counts']}",
        f"- Feature configuration: {result['config']}",
        "",
        "## Model selection",
        "",
        f"Winner selected only by validation macro-F1: **{winner['candidate_id']}** ({winner['validation_macro_f1']:.3f}).",
        "",
        "```text\n" + result["candidates"].to_string(index=False) + "\n```",
        "",
        "## Held-out test results",
        "",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- Macro precision: {metrics['macro_precision']:.3f}",
        f"- Macro recall: {metrics['macro_recall']:.3f}",
        f"- Macro F1: {metrics['macro_f1']:.3f}",
        f"- Weighted F1: {metrics['weighted_f1']:.3f}",
        f"- Winner fit time: {result['test_fit_seconds']:.3f}s",
        f"- Prediction latency: {result['test_prediction_ms_per_sample']:.3f} ms/sample",
        "",
        "### Per-author test results",
        "",
        "```text\n" + metrics["per_author"].to_string(index=False) + "\n```",
        "",
        "### Confusion matrix",
        "",
        "Rows are actual authors; columns are predicted authors in this order: " + ", ".join(map(str, metrics["labels"])) + ".",
        "",
        str(metrics["confusion_matrix"]),
        "",
        "### Commonly confused author pairs",
        "",
        (pd.DataFrame(result["common_confusions"]).to_string(index=False) if result["common_confusions"] else "No off-diagonal test errors."),
        "",
        "## Interpretation limits",
        "",
        "Performance depends on corpus size, genre, topic, time period, editing, quotations, and document leakage. Test scores describe only this held-out sample and should not be generalized without external validation. Feature rankings show model associations, not causal or identity evidence.",
    ]
    lines.extend(["", f"Misclassified test passages: {len(result['test_misclassifications'])}. Correctly classified test passages: {len(result['test_correct_classifications'])}."])
    if include_error_text and result["test_misclassifications"]:
        lines.extend(["", "## Misclassified test passages (text excerpts)", "", "Text excerpts included by user choice; review licensing and privacy before sharing."])
        for item in result["test_misclassifications"][:30]:
            lines.append(f"- Actual **{item['actual']}**, predicted **{item['predicted']}**: {item['text']}")
    if robustness:
        lines.extend(["", "## Robustness and generalization checks", ""])
        lines.append("These comparisons are exploratory. Grouped performance changes can reflect leakage or distribution shift; the metrics do not identify which explanation applies.")
        for title, table in robustness.items():
            lines.extend(["", f"### {title}", "", "```text", table.to_string(index=False), "```"])
    return "\n".join(lines)
