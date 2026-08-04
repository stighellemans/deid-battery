from pathlib import Path

from deid_battery.pseudonymization_eval import run


def test_disabled_pseudonymization_is_a_noop(tmp_path: Path):
    assert run(
        {"evaluate": {"pseudonymization": {"enabled": False}}},
        base_dir=tmp_path,
    ) is None


def test_missing_pseudonymization_dependency_is_not_loaded_when_disabled(tmp_path: Path):
    assert run({"evaluate": {}}, base_dir=tmp_path) is None
