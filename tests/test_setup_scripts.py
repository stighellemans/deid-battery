from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_setup_help_documents_deid_schema_option():
    for script in ("scripts/setup.sh", "scripts/setup_deidentify_venv.sh"):
        result = subprocess.run(
            ["bash", script, "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--deid-schema" in result.stdout


def test_schema_helper_resolves_valid_checkout(tmp_path):
    checkout = tmp_path / "deid-schema"
    (checkout / "src/deid_schema").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

    result = run_bash(
        f"source scripts/_deid_schema.sh; "
        f"resolve_deid_schema_dir {ROOT!s} {checkout!s}"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == str(checkout.resolve())


def test_schema_helper_fails_before_setup_for_invalid_checkout(tmp_path):
    missing = tmp_path / "missing"
    result = run_bash(
        f"source scripts/_deid_schema.sh; "
        f"resolve_deid_schema_dir {ROOT!s} {missing!s}"
    )

    assert result.returncode == 2
    assert "does not exist" in result.stderr


@pytest.mark.parametrize(
    ("script", "extra_args"),
    [
        ("scripts/setup.sh", ["--venv"]),
        ("scripts/setup_deidentify_venv.sh", []),
    ],
)
def test_setup_fails_before_creating_environment_when_schema_is_missing(
    tmp_path, script, extra_args
):
    missing = tmp_path / "missing-schema"
    environment = tmp_path / "should-not-exist"
    args = ["bash", script, "--deid-schema", str(missing), *extra_args, str(environment)]

    result = subprocess.run(
        args,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert not environment.exists()
