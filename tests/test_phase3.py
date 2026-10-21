import numpy as np
import pandas as pd
import pytest

from src.dataset_manager import (
    DatasetValidationError,
    synthetic_demo_dataset,
    train_validation_test_split,
    validate_dataset,
)
from src.comparison import compare_texts_with_fitted_pipeline
from src.explainability import top_model_features
from src.feature_pipeline import make_feature_transformer
from src.persistence import load_saved_model, save_model
from src.training import run_experiments
from src.reporting import experiment_report
from src.robustness import (
    compare_random_and_group_splits,
    error_analysis,
    evaluate_document_length_robustness,
    evaluate_domain_robustness,
    run_feature_ablation,
    run_preprocessing_sensitivity,
)


def test_dataset_validation_and_empty_rows():
    frame = pd.DataFrame({"text": ["  ", "A full sentence.", "A   full sentence.", None], "author": ["A", "A", "A", "B"]})
    cleaned, quality = validate_dataset(frame)
    assert len(cleaned) == 2
    assert quality["removed_empty_text_rows"]
    assert quality["exact_duplicate_rows"] == [1, 2]
    with pytest.raises(DatasetValidationError, match="Missing required"):
        validate_dataset(pd.DataFrame({"text": ["text"]}))


def test_group_split_is_deterministic_and_disjoint():
    frame, _ = validate_dataset(synthetic_demo_dataset())
    first = train_validation_test_split(frame, group_column="document_id", random_seed=13)
    second = train_validation_test_split(frame, group_column="document_id", random_seed=13)
    assert first["sizes"] == second["sizes"]
    assert first["groups"] == second["groups"]
    sets = [set(groups) for groups in first["groups"].values()]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]


def test_group_split_fails_for_too_few_source_groups():
    frame = pd.DataFrame({"text": [f"Some passage {i}." for i in range(12)], "author": ["A"] * 6 + ["B"] * 6, "document_id": ["doc1"] * 6 + ["doc2"] * 6})
    with pytest.raises(DatasetValidationError, match="at least three distinct groups"):
        train_validation_test_split(frame, group_column="document_id", minimum_samples_per_author=3)


def test_style_pipeline_is_nonnegative_and_exposes_names():
    texts = ["A brief sample with several words.", "Another sample has different words and punctuation!"]
    transformer = make_feature_transformer("stylometric")
    matrix = transformer.fit_transform(texts)
    assert matrix.shape[0] == 2
    assert np.all(matrix >= 0)
    assert len(transformer.get_feature_names_out()) == matrix.shape[1]


def test_experiment_selects_by_validation_and_scores_test():
    frame, _ = validate_dataset(synthetic_demo_dataset())
    result = run_experiments(
        frame, representations=["word_unigram_bigram", "stylometric"],
        models=["MultinomialNB", "LinearSVC"], group_column="document_id",
        minimum_samples_per_author=5, max_vocabulary=250, c_values=(0.1,), nb_alphas=(0.5,),
    )
    assert result["winner_id"] == result["candidates"].iloc[0]["candidate_id"]
    assert result["winner"]["validation_macro_f1"] == result["candidates"].iloc[0]["validation_macro_f1"]
    assert result["test_metrics"]["confusion_matrix"].shape == (3, 3)
    assert len(result["test_truth"]) == result["split"]["sizes"]["test"]
    assert top_model_features(result["winner_pipeline"], limit=2)


def test_persisted_local_model_predicts_after_reload(monkeypatch):
    import tempfile
    from pathlib import Path
    import src.persistence as persistence

    frame, _ = validate_dataset(synthetic_demo_dataset())
    result = run_experiments(
        frame, representations=["word_unigram"], models=["MultinomialNB"],
        group_column="document_id", minimum_samples_per_author=5,
        max_vocabulary=200, nb_alphas=(1.0,),
    )
    with tempfile.TemporaryDirectory(prefix=".tmp-model-test-", dir=Path.cwd()) as temporary:
        monkeypatch.setattr(persistence, "ARTIFACT_DIR", Path(temporary))
        manifest = save_model(result["winner_pipeline"], metadata={"winner": result["winner"]}, training_rows=result["train_rows"])
        reloaded, loaded_manifest = load_saved_model(manifest["model_id"])
        assert loaded_manifest["model_id"] == manifest["model_id"]
        prediction = reloaded.predict(["A new sentence from the archive."])
        assert len(prediction) == 1
        with pytest.raises(ValueError):
            load_saved_model("../outside")


def _small_grouped_experiment():
    frame, _ = validate_dataset(synthetic_demo_dataset())
    result = run_experiments(
        frame, representations=["character"], models=["LinearSVC"],
        group_column="document_id", minimum_samples_per_author=5,
        max_vocabulary=250, c_values=(0.1,), nb_alphas=(1.0,),
    )
    kwargs = {
        "test_size": 0.2, "validation_size": 0.2, "random_seed": 42,
        "group_column": "document_id", "minimum_samples_per_author": 5,
        "max_vocabulary": 200, "c_values": (0.1,), "nb_alphas": (1.0,),
    }
    return frame, result, kwargs


def test_fitted_pipeline_comparison_and_error_analysis():
    _, result, _ = _small_grouped_experiment()
    compared = compare_texts_with_fitted_pipeline(result["winner_pipeline"], "One quiet page.", "Another different note.")
    assert compared["feature_count"] > 0
    assert compared["prediction_a"] in result["authors"]
    errors = error_analysis(result)
    assert len(errors["correct_examples"]) + len(errors["misclassified_examples"]) == result["split"]["sizes"]["test"]
    assert "count" in errors["common_confusions"].columns


def test_document_length_robustness_reports_counts_and_small_groups():
    _, result, _ = _small_grouped_experiment()
    evaluated = evaluate_document_length_robustness(result, thresholds=(0, 50, 100), minimum_group_samples=10)
    assert evaluated["thresholds"].iloc[0]["samples"] == result["split"]["sizes"]["test"]
    assert evaluated["thresholds"].iloc[-1]["samples"] == 0
    assert evaluated["length_groups"]["samples"].sum() == result["split"]["sizes"]["test"]
    assert evaluated["thresholds"].iloc[-1]["warning"]


def test_domain_robustness_reports_sample_counts_and_low_count_warning():
    _, result, _ = _small_grouped_experiment()
    # Attach domain metadata to the same held-out rows and predictions.
    result["test_rows"] = result["test_rows"].copy()
    result["test_rows"]["genre"] = ["narrative" if index % 2 else "notes" for index in range(len(result["test_rows"]))]
    domain = evaluate_domain_robustness(result, column="genre", minimum_group_samples=10)
    assert domain["summary"]["samples"].sum() == result["split"]["sizes"]["test"]
    assert domain["summary"]["warning"].notna().all()


def test_split_comparison_ablation_and_sensitivity_use_reproducible_protocols():
    frame, _, kwargs = _small_grouped_experiment()
    splits = compare_random_and_group_splits(frame, group_column="document_id", run_kwargs={**kwargs, "representations": ["character"], "models": ["LinearSVC"]})
    assert len(splits["summary"]) == 2
    assert "does not identify the cause" in splits["interpretation"]
    ablation = run_feature_ablation(frame, run_kwargs=kwargs, model="LinearSVC")
    assert set(ablation["summary"]["representation"]) == {"stylometric", "word_unigram_bigram", "character", "hybrid_word"}
    sensitivity = run_preprocessing_sensitivity(frame, run_kwargs=kwargs, model="LinearSVC")
    assert len(sensitivity["summary"]) == 6
    assert sensitivity["summary"]["test_samples"].nunique() == 1


def test_report_omits_raw_misclassified_text_by_default():
    _, result, _ = _small_grouped_experiment()
    result["test_misclassifications"] = [{"actual": "A", "predicted": "B", "text": "private excerpt"}]
    result["common_confusions"] = [{"actual": "A", "predicted": "B", "count": 1}]
    result["test_correct_classifications"] = []
    report = experiment_report(result)
    assert "private excerpt" not in report
    assert "Misclassified test passages: 1" in report
    assert "private excerpt" in experiment_report(result, include_error_text=True)
