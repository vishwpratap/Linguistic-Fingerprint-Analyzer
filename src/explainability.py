"""Model coefficient summaries and validation/test error inspection."""

from __future__ import annotations

from typing import Any

import numpy as np


def top_model_features(pipeline: Any, limit: int = 12) -> list[dict[str, Any]]:
    """Return the strongest signed class features for linear models or NB."""
    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    names = np.asarray(features.get_feature_names_out(), dtype=object)
    output = []
    classes = list(classifier.classes_)
    if hasattr(classifier, "coef_"):
        weights = np.asarray(classifier.coef_)
        if len(classes) == 2 and weights.shape[0] == 1:
            weights = np.vstack((-weights[0], weights[0]))
        for class_index, author in enumerate(classes):
            row = weights[class_index]
            for index in np.argsort(row)[-limit:][::-1]:
                output.append({"author": author, "direction": "supports", "feature": str(names[index]), "weight": float(row[index])})
            for index in np.argsort(row)[:limit]:
                output.append({"author": author, "direction": "argues against", "feature": str(names[index]), "weight": float(row[index])})
    elif hasattr(classifier, "feature_log_prob_"):
        log_prob = np.asarray(classifier.feature_log_prob_)
        for class_index, author in enumerate(classes):
            own = log_prob[class_index]
            others = np.mean(np.delete(log_prob, class_index, axis=0), axis=0) if len(classes) > 1 else np.zeros_like(own)
            delta = own - others
            for index in np.argsort(delta)[-limit:][::-1]:
                output.append({"author": author, "direction": "more likely in class", "feature": str(names[index]), "weight": float(delta[index])})
    else:
        return []
    return output

