#!/usr/bin/env python3
"""Fail fast when a hospital battery checkout is dirty, unpinned, or incomplete."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "deployment" / "hospital-source-lock.json"


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _resolve(value: str, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _load_mapping(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(config_path: Path, lock_path: Path, *, code_only: bool, allow_dirty: bool) -> list[str]:
    errors: list[str] = []
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    config = _load_mapping(config_path)

    if not allow_dirty:
        status = _git_output(ROOT, "status", "--porcelain", "--untracked-files=all")
        if status:
            errors.append("deid-battery checkout has uncommitted or untracked files")

    for name, source in lock["git_sources"].items():
        if not source.get("required_checkout"):
            continue
        repo = _resolve(source["path"])
        if not (repo / ".git").exists():
            errors.append(f"required checkout missing: {name} at {repo}")
            continue
        actual = _git_output(repo, "rev-parse", "HEAD")
        if actual != source["commit"]:
            errors.append(f"{name} is at {actual or 'UNKNOWN'}, expected {source['commit']}")
        if not allow_dirty and _git_output(repo, "status", "--porcelain", "--untracked-files=all"):
            errors.append(f"{name} checkout has uncommitted or untracked files")

    pinned_models = lock["huggingface_models"]
    for model in config.get("models", []):
        params = model.get("params") or {}
        model_name = params.get("model")
        runner = model.get("runner")
        if runner in {"gliner", "hf_token"} and model_name in pinned_models:
            if params.get("revision") != pinned_models[model_name]:
                errors.append(f"{model['id']} does not use the locked revision for {model_name}")
        if runner == "robbert":
            base_model = params.get("base_model")
            if base_model in pinned_models and params.get("base_revision") != pinned_models[base_model]:
                errors.append(f"{model['id']} does not use the locked revision for {base_model}")
        if runner == "llm":
            hostname = urlparse(str(params.get("base_url") or "")).hostname
            if hostname not in {"127.0.0.1", "localhost", "::1"}:
                errors.append(
                    f"{model['id']} LLM endpoint is not loopback-only: {params.get('base_url')!r}"
                )

    if code_only:
        return errors

    input_path = _resolve(str(config.get("input", "")))
    if not input_path.is_file():
        errors.append(f"missing input: {input_path}")
    bundle = _resolve(str((config.get("evaluate") or {}).get("bundle", "")))
    for filename in ("manifest.json", "reference_items.jsonl"):
        if not (bundle / filename).is_file():
            errors.append(f"missing evaluation bundle file: {bundle / filename}")

    for model in config.get("models", []):
        params = model.get("params") or {}
        if params.get("checkpoint"):
            checkpoint = _resolve(str(params["checkpoint"]))
            if not checkpoint.is_file():
                errors.append(f"missing checkpoint for {model['id']}: {checkpoint}")
        if model.get("venv"):
            python = _resolve(str(model["venv"])) / "bin" / "python"
            if not python.is_file():
                errors.append(f"missing worker environment for {model['id']}: {python}")
        if params.get("prompt_dir"):
            prompt_dir = _resolve(str(params["prompt_dir"]))
            for filename in ("dict_prompt.txt", "dict_example.txt", "labels.csv"):
                if not (prompt_dir / filename).is_file():
                    errors.append(f"missing prompt asset for {model['id']}: {prompt_dir / filename}")
        if params.get("model_file") or params.get("model_sha256_file"):
            model_file = _resolve(str(params.get("model_file") or ""))
            checksum_file = _resolve(str(params.get("model_sha256_file") or ""))
            if not model_file.is_file():
                errors.append(f"missing local model for {model['id']}: {model_file}")
            if not checksum_file.is_file():
                errors.append(f"missing model checksum for {model['id']}: {checksum_file}")
            if model_file.is_file() and checksum_file.is_file():
                expected = checksum_file.read_text(encoding="utf-8").split()[0].lower()
                actual = _sha256(model_file)
                if expected != actual:
                    errors.append(
                        f"model checksum mismatch for {model['id']}: {actual} != {expected}"
                    )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/battery.vm.yaml")
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--code-only", action="store_true", help="skip data, model, and venv checks")
    parser.add_argument("--allow-dirty", action="store_true", help="permit local source edits")
    args = parser.parse_args()

    errors = check(
        _resolve(args.config),
        _resolve(args.lock),
        code_only=args.code_only,
        allow_dirty=args.allow_dirty,
    )
    if errors:
        print("hospital preflight FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("hospital preflight OK: sources, revisions, endpoint policy, and required assets match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
