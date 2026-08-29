---
description: Rules for tests
globs: tests/**
---
# Test rules
- `tests/unit` runs on CPU. Mark tests needing torchtitan with `@pytest.mark.titan`, NPU with
  `@pytest.mark.npu`; conftest skips them when unavailable.
- Use the `clean_registry` fixture for shim tests; never rely on real shims being present.
- Integration runs on NPU use upstream's `tests/integration_tests/run_tests.py` from the pinned
  checkout (clone, don't vendor).
