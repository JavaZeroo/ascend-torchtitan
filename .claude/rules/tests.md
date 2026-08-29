---
description: 测试规则
globs: tests/**
---
# 测试规则
- `tests/unit` 在 CPU 上跑。需要 torchtitan 的测试标 `@pytest.mark.titan`，需要 NPU 的标 `@pytest.mark.npu`；conftest 在缺失时跳过。
- shim 测试用 `clean_registry` fixture；绝不依赖真实 shim 已存在。需要真实 shim 模块时先 `importlib.reload` 让装饰器重新注册。
- NPU 上的集成运行用 `ascend_titan.tools.matrix`（复用固定检出中上游的 `tests/integration_tests` 用例；克隆，不 vendor）。
