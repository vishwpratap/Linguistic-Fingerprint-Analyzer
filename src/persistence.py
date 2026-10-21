"""Controlled local persistence for app-generated trusted model artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / ".artifacts" / "authorship_models"
MODEL_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def dataset_fingerprint(frame: Any) -> str:
    """Hash normalized training text/labels for provenance without storing raw text."""
    digest = hashlib.sha256()
    for text, author in frame[["text", "author"]].itertuples(index=False, name=None):
        digest.update(str(author).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def save_model(pipeline: Any, *, metadata: dict[str, Any], training_rows: Any) -> dict[str, Any]:
    """Save an app-generated sklearn pipeline under an opaque random id."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_id = uuid.uuid4().hex
    model_path = ARTIFACT_DIR / f"{model_id}.joblib"
    manifest_path = ARTIFACT_DIR / f"{model_id}.json"
    manifest = {
        **metadata,
        "model_id": model_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(training_rows)),
        "training_data_fingerprint_sha256": dataset_fingerprint(training_rows),
        "artifact_format": "joblib sklearn pipeline; trusted files created by this app only",
    }
    # The user-facing app never loads arbitrary uploads or path strings.
    joblib.dump(pipeline, model_path, compress=3)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def list_saved_models() -> list[dict[str, Any]]:
    """List model IDs with valid manifests; never returns paths from user input."""
    if not ARTIFACT_DIR.exists():
        return []
    output = []
    for manifest_path in ARTIFACT_DIR.glob("*.json"):
        model_id = manifest_path.stem
        model_path = ARTIFACT_DIR / f"{model_id}.joblib"
        if not MODEL_ID_PATTERN.fullmatch(model_id) or not model_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("model_id") == model_id:
                output.append(manifest)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(output, key=lambda item: item.get("created_utc", ""), reverse=True)


def load_saved_model(model_id: str) -> tuple[Any, dict[str, Any]]:
    """Load only an opaque app-generated model ID from the controlled directory."""
    if not MODEL_ID_PATTERN.fullmatch(str(model_id)):
        raise ValueError("Invalid saved model id.")
    model_path = (ARTIFACT_DIR / f"{model_id}.joblib").resolve()
    manifest_path = (ARTIFACT_DIR / f"{model_id}.json").resolve()
    root = ARTIFACT_DIR.resolve()
    if model_path.parent != root or manifest_path.parent != root or not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Saved model was not found in the local model store.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != model_id:
        raise ValueError("Saved model manifest does not match its id.")
    return joblib.load(model_path), manifest

