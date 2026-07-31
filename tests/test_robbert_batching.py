from types import SimpleNamespace

from deid_battery.runners.robbert import _is_out_of_memory, _resolved_batch_size


def test_automatic_batch_size_by_device():
    assert _resolved_batch_size(0, SimpleNamespace(type="cuda")) == 16
    assert _resolved_batch_size(None, SimpleNamespace(type="mps")) == 8
    assert _resolved_batch_size(0, SimpleNamespace(type="cpu")) == 1


def test_explicit_batch_size_overrides_device_default():
    assert _resolved_batch_size(3, SimpleNamespace(type="cuda")) == 3
    assert _resolved_batch_size(3, SimpleNamespace(type="mps")) == 3


def test_cuda_and_mps_oom_messages_are_recognised():
    assert _is_out_of_memory(RuntimeError("CUDA out of memory"))
    assert _is_out_of_memory(RuntimeError("MPS backend out of memory"))
    assert not _is_out_of_memory(RuntimeError("checkpoint mismatch"))
