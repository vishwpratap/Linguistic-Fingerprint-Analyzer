"""Dataset validation, descriptive quality checks, and leakage-aware splitting."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .preprocessing import normalize_text

REQUIRED_COLUMNS = ("text", "author")
OPTIONAL_GROUP_COLUMNS = ("document_id", "source")


class DatasetValidationError(ValueError):
    """Raised when an authorship dataset does not meet its declared schema."""


def validate_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate schema, remove empty text/missing labels, and report duplicates.

    Labels are kept exactly as supplied; whitespace and capitalization are not
    normalized, so distinct labels are never silently merged.
    """
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise DatasetValidationError(f"Missing required column(s): {', '.join(missing_columns)}. Expected text and author.")
    cleaned = frame.copy()
    original_rows = len(cleaned)
    text_missing = cleaned["text"].isna() | cleaned["text"].astype(str).str.strip().eq("")
    author_missing = cleaned["author"].isna() | cleaned["author"].astype(str).str.strip().eq("")
    empty_text_rows = cleaned.index[text_missing].tolist()
    missing_author_rows = cleaned.index[author_missing].tolist()
    cleaned = cleaned.loc[~(text_missing | author_missing)].copy()
    cleaned["text"] = cleaned["text"].astype(str)
    if cleaned.empty:
        raise DatasetValidationError("No usable rows remain after removing empty texts and missing author labels.")

    # Mixed scalar label types make scikit-learn classification ambiguous.
    label_types = {type(label) for label in cleaned["author"].tolist()}
    if len(label_types) > 1:
        raise DatasetValidationError("Author labels contain mixed data types. Make the author column consistently text or consistently numeric; labels are not coerced or merged automatically.")

    canonical = cleaned["text"].map(lambda value: normalize_text(value).casefold())
    duplicate_mask = canonical.duplicated(keep=False)
    exact_duplicate_rows = cleaned.index[duplicate_mask].tolist()
    groups = []
    for value, group in cleaned.loc[duplicate_mask].assign(_canonical=canonical[duplicate_mask]).groupby("_canonical", sort=False):
        groups.append({"rows": group.index.tolist(), "authors": group["author"].tolist()})

    counts = cleaned["author"].value_counts(dropna=False).to_dict()
    count_values = list(counts.values())
    imbalance_ratio = max(count_values) / min(count_values) if count_values and min(count_values) else None
    warnings = []
    if exact_duplicate_rows:
        warnings.append(f"Found {len(exact_duplicate_rows)} rows participating in exact/whitespace-normalized duplicates; inspect before training to avoid leakage or repeated-example bias.")
    group_missing = {}
    for column in OPTIONAL_GROUP_COLUMNS:
        if column in cleaned:
            missing = int(cleaned[column].isna().sum() + cleaned[column].astype(str).str.strip().eq("").sum())
            group_missing[column] = missing
            if missing:
                warnings.append(f"{missing} rows have missing {column} values; grouped splitting cannot use this column until they are fixed or removed.")
    if not any(column in cleaned for column in OPTIONAL_GROUP_COLUMNS):
        warnings.append("No document_id or source column is available. Random passage-level splits may overestimate generalization if passages share an original source.")
    quality = {
        "original_rows": original_rows,
        "usable_rows": len(cleaned),
        "removed_empty_text_rows": empty_text_rows,
        "removed_missing_author_rows": missing_author_rows,
        "authors": len(counts),
        "class_counts": counts,
        "imbalance_ratio": imbalance_ratio,
        "exact_duplicate_rows": exact_duplicate_rows,
        "exact_duplicate_groups": groups,
        "group_missing_counts": group_missing,
        "warnings": warnings,
    }
    return cleaned.reset_index(drop=True), quality


def find_near_duplicates(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.92,
    max_samples: int = 500,
    max_comparisons: int = 5000,
) -> dict[str, Any]:
    """Find a bounded, deterministic set of similar normalized text pairs.

    Exact duplicates are omitted. Near-duplicate checks are intentionally
    capped so exploratory dataset inspection stays responsive.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be within (0, 1]")
    if len(frame) > max_samples:
        return {"pairs": [], "comparisons": 0, "skipped": True, "warning": f"Near-duplicate check skipped above {max_samples} rows; exact duplicates were still checked."}
    texts = [normalize_text(value).casefold() for value in frame["text"].astype(str)]
    pairs = []
    comparisons = 0
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            if texts[left] == texts[right]:
                continue
            if comparisons >= max_comparisons:
                return {"pairs": pairs, "comparisons": comparisons, "skipped": True, "warning": f"Near-duplicate check stopped at its {max_comparisons} comparison cap."}
            comparisons += 1
            ratio = SequenceMatcher(None, texts[left][:5000], texts[right][:5000], autojunk=False).ratio()
            if ratio >= threshold:
                pairs.append({
                    "row_a": frame.index[left], "row_b": frame.index[right], "similarity": ratio,
                    "author_a": frame.iloc[left]["author"], "author_b": frame.iloc[right]["author"],
                })
    return {"pairs": pairs, "comparisons": comparisons, "skipped": False, "warning": None}


def eligible_authors(frame: pd.DataFrame, minimum_samples: int = 5) -> list[Any]:
    """Return labels with at least the configured number of samples."""
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be at least 1")
    counts = frame["author"].value_counts()
    return [label for label, count in counts.items() if count >= minimum_samples]


def filter_authors(frame: pd.DataFrame, authors: list[Any] | None = None, minimum_samples: int = 5) -> pd.DataFrame:
    """Filter to selected eligible labels without normalizing their names."""
    allowed = set(eligible_authors(frame, minimum_samples) if authors is None else authors)
    eligible = set(eligible_authors(frame, minimum_samples))
    if not allowed.issubset(eligible):
        raise DatasetValidationError("Selected authors must each meet the configured minimum sample count.")
    return frame[frame["author"].isin(allowed)].copy().reset_index(drop=True)


def train_validation_test_split(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_seed: int = 42,
    group_column: str | None = None,
    minimum_samples_per_author: int = 5,
    max_group_attempts: int = 100,
) -> dict[str, Any]:
    """Create deterministic stratified 60/20/20 splits or group-disjoint splits.

    Test-set size and validation-set size are fractions of the full dataset.
    Grouped splits retry deterministic holdouts and fail clearly if every class
    cannot be represented in all three partitions without source leakage.
    """
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive fractions whose sum is less than one")
    labels = frame["author"].to_numpy()
    counts = frame["author"].value_counts()
    rare = counts[counts < minimum_samples_per_author]
    if not rare.empty:
        details = ", ".join(f"{label}: {count}" for label, count in rare.items())
        raise DatasetValidationError(f"Each author needs at least {minimum_samples_per_author} samples before splitting. Insufficient: {details}.")
    all_labels = set(frame["author"].tolist())
    indices = np.arange(len(frame))
    relative_validation_size = validation_size / (1 - test_size)

    if group_column is None:
        try:
            train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_seed, stratify=labels)
            train_idx, validation_idx = train_test_split(
                train_val_idx,
                test_size=relative_validation_size,
                random_state=random_seed + 1,
                stratify=labels[train_val_idx],
            )
        except ValueError as error:
            raise DatasetValidationError(
                "The requested stratified three-way split is too small for these class counts and split proportions. Add examples per author or adjust the split sizes. Details: " + str(error)
            ) from error
        method = "two-stage stratified random split"
        groups_used = None
    else:
        if group_column not in frame:
            raise DatasetValidationError(f"Group column '{group_column}' is not present in the dataset.")
        group_values = frame[group_column]
        if group_values.isna().any() or group_values.astype(str).str.strip().eq("").any():
            raise DatasetValidationError(f"Group column '{group_column}' contains missing values; grouped splitting would be unsafe.")
        group_array = group_values.to_numpy()
        per_author_groups = frame.groupby("author", dropna=False)[group_column].nunique()
        if (per_author_groups < 3).any():
            low = ", ".join(str(label) for label in per_author_groups[per_author_groups < 3].index)
            raise DatasetValidationError(f"Group-aware three-way splitting needs at least three distinct groups per author. Insufficient: {low}.")
        found = None
        for attempt in range(max_group_attempts):
            test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed + attempt)
            try:
                train_val_idx, candidate_test_idx = next(test_splitter.split(indices, labels, group_array))
                validation_splitter = GroupShuffleSplit(n_splits=1, test_size=relative_validation_size, random_state=random_seed + 10_000 + attempt)
                candidate_train_rel, candidate_val_rel = next(
                    validation_splitter.split(train_val_idx, labels[train_val_idx], group_array[train_val_idx])
                )
            except ValueError:
                continue
            candidate_train_idx = train_val_idx[candidate_train_rel]
            candidate_validation_idx = train_val_idx[candidate_val_rel]
            partitions = (candidate_train_idx, candidate_validation_idx, candidate_test_idx)
            if all(set(labels[part].tolist()) == all_labels for part in partitions):
                found = partitions
                break
        if found is None:
            raise DatasetValidationError(
                f"Could not create three group-disjoint partitions with every author represented after {max_group_attempts} deterministic attempts. Add source documents or revise the split proportions."
            )
        train_idx, validation_idx, test_idx = found
        method = f"two-stage GroupShuffleSplit grouped by {group_column} (deterministic retries)"
        groups_used = {
            "train": sorted(set(group_array[train_idx].tolist()), key=str),
            "validation": sorted(set(group_array[validation_idx].tolist()), key=str),
            "test": sorted(set(group_array[test_idx].tolist()), key=str),
        }

    def class_counts(partition: np.ndarray) -> dict[Any, int]:
        return frame.iloc[partition]["author"].value_counts().to_dict()

    return {
        "train_indices": np.asarray(train_idx),
        "validation_indices": np.asarray(validation_idx),
        "test_indices": np.asarray(test_idx),
        "method": method,
        "group_column": group_column,
        "groups": groups_used,
        "class_counts": {
            "train": class_counts(train_idx),
            "validation": class_counts(validation_idx),
            "test": class_counts(test_idx),
        },
        "sizes": {"train": len(train_idx), "validation": len(validation_idx), "test": len(test_idx)},
        "random_seed": random_seed,
        "test_size": test_size,
        "validation_size": validation_size,
    }


def synthetic_demo_dataset() -> pd.DataFrame:
    """Create fictional, deliberately patterned data for functionality checks."""
    topics = (
        "the station clock", "a winter garden", "the harbor ledger", "an empty classroom",
        "the glasshouse path", "a brass key", "the western footbridge", "a paper map",
        "the market bell", "a flooded field", "the blue envelope", "a quiet observatory",
    )
    times = ("before sunrise", "at midday", "just after rain", "near closing time")
    details = ("a narrow scratch", "three late arrivals", "a missing date", "one pale reflection", "a folded corner", "an uneven step", "a quiet signal", "a faint blue mark", "a change in temperature", "two open windows", "a line of salt", "a distant whistle")
    obstacles = ("the wind kept changing", "the ink had nearly dried", "a door would not stay open", "the path was covered in leaves", "the room grew noisy", "the light kept moving", "the page was torn", "the bell rang again", "the rain returned", "the signal faded", "the floor was uneven", "the glass had fogged over")
    descriptions = ("useful", "unexpected", "uncertain", "plain", "important", "curious", "hard to explain", "easy to overlook", "worth recording", "strangely familiar", "unfinished", "clear enough")
    rows = []
    for author_index, author in enumerate(("Synthetic Author A", "Synthetic Author B", "Synthetic Author C")):
        for book_index in range(1, 5):
            for passage_index in range(1, 4):
                index = (book_index - 1) * 3 + passage_index - 1
                topic, time, detail, obstacle, description = topics[index], times[index % len(times)], details[index], obstacles[index], descriptions[index]
                if author_index == 0:
                    text = f"At {time}, I examined {topic}. I recorded {detail} in the margin and checked the sequence carefully. Although {obstacle}, the observation remained {description}. I noted why it mattered and preserved the order of every entry."
                elif author_index == 1:
                    text = f"{time}. {topic}, again. {detail}. I checked once. Then twice. {obstacle}—so I stopped. The note says {description}. One line crossed out. Page shut. That was enough."
                else:
                    text = f"What changed at {topic} {time}? I noticed {detail}; perhaps it was {description}. If {obstacle}, would the result look different? I wrote the question down, watched again, and waited for an answer. Who else would notice?"
                text += f" This generated study passage {passage_index} from fictional notebook {book_index} contains no historical quotation or real person's work."
                rows.append({
                    "text": text,
                    "author": author,
                    "document_id": f"{author.replace(' ', '_')}_fiction_{book_index}",
                    "source": "Synthetic demonstration generated in-app",
                    "genre": "fictional notebook passage",
                })
    return pd.DataFrame(rows)
