from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts.preflight_hospital import check


ROOT = Path(__file__).resolve().parents[1]


def _lock(tmp_path: Path) -> Path:
    value = json.loads((ROOT / "deployment/hospital-source-lock.json").read_text())
    value["git_sources"] = {}
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_preflight_accepts_locked_loopback_models(tmp_path):
    config = {
        "models": [
            {
                "id": "gliner",
                "runner": "gliner",
                "params": {
                    "model": "urchade/gliner_multi_pii-v1",
                    "model_commit": "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d",
                },
            },
            {
                "id": "qwen",
                "runner": "llm",
                "params": {"base_url": "http://127.0.0.1:8089/v1"},
            },
        ]
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert check(config_path, _lock(tmp_path), code_only=True, allow_dirty=True) == []


def test_preflight_accepts_legacy_revision_name(tmp_path):
    config = {
        "models": [
            {
                "id": "gliner",
                "runner": "gliner",
                "params": {
                    "model": "urchade/gliner_multi_pii-v1",
                    "revision": "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d",
                },
            }
        ]
    }
    config_path = tmp_path / "legacy-config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert check(config_path, _lock(tmp_path), code_only=True, allow_dirty=True) == []


def test_preflight_rejects_missing_model_commit_and_external_llm(tmp_path):
    config = {
        "models": [
            {
                "id": "gliner",
                "runner": "gliner",
                "params": {"model": "urchade/gliner_multi_pii-v1"},
            },
            {
                "id": "qwen",
                "runner": "llm",
                "params": {"base_url": "https://example.test/v1"},
            },
        ]
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    errors = check(config_path, _lock(tmp_path), code_only=True, allow_dirty=True)

    assert any("locked model commit" in error for error in errors)
    assert any("loopback-only" in error for error in errors)


def test_canonical_config_uses_locked_models_and_local_llm(tmp_path):
    errors = check(
        ROOT / "configs/battery.yaml",
        _lock(tmp_path),
        code_only=True,
        allow_dirty=True,
    )

    assert errors == []


def _model(config: dict, model_id: str) -> dict:
    return next(model for model in config["models"] if model["id"] == model_id)


def test_canonical_qwen_keeps_validated_hospital_settings():
    config = yaml.safe_load((ROOT / "configs/battery.yaml").read_text())
    qwen = _model(config, "qwen3-8b")

    assert qwen["device_label"] == "gpu"
    assert qwen["params"] == {
        "base_url": "http://127.0.0.1:11500/v1",
        "model": "qwen3:8b",
        "prompt_dir": "prompts",
        "temperature": 0.6,
        "top_p": 0.95,
        "thinking": True,
        "max_tokens": 8000,
        "workers": 2,
    }


def test_hospital_deidentify_keeps_established_runtime_and_memory_guards():
    config = yaml.safe_load((ROOT / "configs/battery.yaml").read_text())
    deidentify = _model(config, "deidentify")

    assert deidentify["venv"] == "/opt/.venv-deidentify"
    assert deidentify["params"] == {
        "model": "model_bilstmcrf_ons_large-v0.2.0",
        "device": "cpu",
        "chunk": 50,
        "max_chars": 20_000,
        "overlap": 500,
    }


def test_full_preflight_checks_locked_artifact_hash(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"validated model")
    model_lock = tmp_path / "models.sha256"
    model_lock.write_text(
        f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  {artifact}\n",
        encoding="utf-8",
    )
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    (bundle / "reference_items.jsonl").write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": str(input_path),
                "evaluate": {"bundle": str(bundle)},
                "models": [],
            }
        ),
        encoding="utf-8",
    )

    assert check(
        config_path,
        _lock(tmp_path),
        code_only=False,
        allow_dirty=True,
        model_lock_path=model_lock,
    ) == []


def test_full_preflight_rejects_missing_train_metrics(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"validated model")
    model_lock = tmp_path / "models.sha256"
    model_lock.write_text(
        f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  {artifact}\n",
        encoding="utf-8",
    )
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    (bundle / "reference_items.jsonl").write_text("{}\n", encoding="utf-8")
    missing_metrics = tmp_path / "train_metrics.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": str(input_path),
                "evaluate": {"bundle": str(bundle)},
                "models": [
                    {
                        "id": "uza",
                        "runner": "robbert",
                        "params": {
                            "checkpoint": str(artifact),
                            "train_metrics": str(missing_metrics),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    errors = check(
        config_path,
        _lock(tmp_path),
        code_only=False,
        allow_dirty=True,
        model_lock_path=model_lock,
    )

    assert errors == [f"missing train metrics for uza: {missing_metrics}"]
