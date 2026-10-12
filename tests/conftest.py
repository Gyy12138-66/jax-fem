"""Lane rules for the test suite (see LANES.md and tests/run_gates.sh).

* ``cpu_platform``: the Kaess G0-v2 approval tests declare the CPU platform and
  compare it with the actual JAX backend; on a machine with a GPU, JAX picks the
  GPU unless JAX_PLATFORMS=cpu (or JAX_PLATFORM_NAME=cpu) is set, and the check
  then fails by design. They are marked here (kaess_2023 is frozen, so its test
  files are not edited) and SKIPPED with the reason when the backend is not CPU.
* The run-manifest date-time test needs jsonschema's ``date-time`` format
  checker (rfc3339-validator, i.e. ``jsonschema[format-nongpl]``); without it the
  format is not checked at all, so the test is skipped with the reason instead
  of failing.
Both run for real in the contract lane: JAX_PLATFORMS=cpu with the
~/miniforge3 jax-fem-env (the pipeline's CPU-side environment).
"""
import os

import pytest

CPU_PLATFORM_MODULES = ("tests/contract/test_kaess_g0v2_approval.py",)
DATETIME_FORMAT_TESTS = ("tests/contract/test_kaess_contracts.py::test_run_manifest_rejects_invalid_datetime_format",)


def _jax_backend():
    forced = (os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME") or "").strip().lower()
    if forced:
        return forced.split(",")[0]
    try:
        import jax
        return jax.default_backend()
    except Exception:  # pragma: no cover - jax missing: nothing to compare
        return "cpu"


def _has_datetime_checker():
    try:
        from jsonschema import FormatChecker
        return "date-time" in FormatChecker.checkers
    except Exception:  # pragma: no cover
        return False


def pytest_collection_modifyitems(config, items):
    cpu_items = [it for it in items if it.nodeid.startswith(CPU_PLATFORM_MODULES)]
    for it in cpu_items:
        it.add_marker(pytest.mark.cpu_platform)
    if cpu_items:
        backend = _jax_backend()
        if backend != "cpu":
            skip = pytest.mark.skip(reason=f"cpu_platform test on JAX backend {backend!r}: run with JAX_PLATFORMS=cpu (contract lane)")
            for it in cpu_items:
                it.add_marker(skip)
    dt_items = [it for it in items if it.nodeid in DATETIME_FORMAT_TESTS]
    if dt_items and not _has_datetime_checker():
        skip = pytest.mark.skip(reason="jsonschema has no date-time format checker: install jsonschema[format-nongpl] "
                                       "or use the contract-lane env (~/miniforge3 jax-fem-env)")
        for it in dt_items:
            it.add_marker(skip)
