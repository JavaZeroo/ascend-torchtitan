---
name: shim-authoring
description: 在 ascend_titan/compat/shims 里编写、测试并注册一条 L0 compat shim。用于失败归因为 TT/TORCH（上游写死 CUDA 或 nightly-only API，且没有配置开关）而 override 够不着的情况。
---
# shim-authoring

## 门槛（全部满足才继续）
- [ ] 归因是 **TT** 或 **TORCH**（不是 NPU/CANN/DEP）。是 NPU 就停：提 torch_npu issue（P1）。
- [ ] 没有配置开关：grep `../torchtitan/torchtitan/config/configs.py` 和相关 `Config` 数据类（P0）。
- [ ] 这段代码不是 `Configurable` 节点（否则写 override，P6）。
- [ ] 已起草上游 issue：`docs/issues/<owner>.md` 里有带锚点的条目，`upstream="draft:docs/issues/<owner>.md#<anchor>"`（P4）；提交后换成 URL。

## 编写
1. `ascend_titan/compat/shims/<name>.py`：
   ```python
   from ascend_titan.compat import shim

   @shim(target="torchtitan.x.y:fn", reason="…", upstream="draft:docs/issues/torchtitan.md#…")
   def <name>(original):
       def wrapped(*a, **k):
           …            # 加 NPU 行为
           return original(*a, **k)   # 总是交回上游（P3）
       return wrapped
   ```
   缺失的 API 用 `kind="polyfill"`（`fn(None)` 返回实现；已存在时自动跳过）。优先包装*小*的辅助函数而不是大函数。若必须 `kind="replace"`，如实写 `why_not_wrap=`，并在 `docs/upstream-tracking.md` 的替换型列表里登记。
2. `tests/unit/test_shim_<name>.py` 里用 `clean_registry` fixture 和一个假目标模块测试：断言原函数被调用且新增行为发生。真实 shim 模块需 `importlib.reload` 后再 `apply_all()`。
3. 在 `docs/upstream-tracking.md` → "shim ↔ 上游 issue" 加一行。
4. 翻转促使你写它的矩阵格子，在归因列注明 shim 名。

## 验证
`pytest tests/unit -x && ascend-titan-doctor`（shim 出现在 "shims registered" 下），然后在 NPU 上跑触发它的用例。
