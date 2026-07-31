from __future__ import annotations

from pathlib import Path

from deid_battery.orchestrate import _runtime_config


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return {
        "input": "input.jsonl",
        "output_dir": "out",
        "timings": "timings.yaml",
        "evaluate": {"enabled": True, "bundle": "evaluation_bundle"},
        "models": [
            {"id": "rules", "runner": "deduce", "params": {"engine": "deduce"}},
            {
                "id": "deidentify",
                "runner": "deidentify",
                "venv": "/opt/.venv-deidentify",
                "params": {"model": "legacy"},
            },
            {
                "id": "qwen",
                "runner": "llm",
                "params": {"base_url": "http://127.0.0.1:11500/v1", "model": "hospital"},
            },
        ],
    }


def test_runtime_overrides_make_one_config_portable_without_mutating_it():
    original = _config()
    resolved = _runtime_config(
        original,
        input_path="input.smoke.jsonl",
        output_dir="out_smoke",
        evaluation_bundle="evaluation_bundle.smoke",
        timings="timings.smoke.yaml",
        llm_base_url="http://127.0.0.1:11434/v1",
        llm_model="qwen3:8b",
        llm_device_label="gpu",
        deidentify_venv=".venv-deidentify",
    )

    assert original == _config()
    assert resolved["input"] == "input.smoke.jsonl"
    assert resolved["output_dir"] == "out_smoke"
    assert resolved["timings"] == "timings.smoke.yaml"
    assert resolved["evaluate"]["bundle"] == "evaluation_bundle.smoke"
    by_id = {model["id"]: model for model in resolved["models"]}
    assert by_id["qwen"]["params"] == {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen3:8b",
    }
    assert by_id["qwen"]["device_label"] == "gpu"
    assert by_id["deidentify"]["venv"] == ".venv-deidentify"


def test_runtime_exclude_removes_models_from_run_and_evaluation():
    resolved = _runtime_config(_config(), exclude=["deidentify", "qwen"])

    assert [model["id"] for model in resolved["models"]] == ["rules"]


def test_repository_has_one_battery_yaml():
    assert list((ROOT / "configs").glob("*.yaml")) == [ROOT / "configs" / "battery.yaml"]
