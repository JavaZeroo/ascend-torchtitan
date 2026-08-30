# patches/ops-nn

ops-nn（AscendC 算子库，含 Kimi-K3 的 SiTU-GLU）的修复。

**和 `patches/torch_npu/`、`patches/op-plugin/` 的区别**：那两个仓在
`gitcode.com/Ascend/*`，属于授权范围，修完要按 P9 提 issue + PR；ops-nn 在
`gitcode.com/cann/ops-nn`，**不在授权范围**（P10），所以这里的补丁只本地应用、
不提 issue，等能提的人来提。

和 `patches/evidence/` 也不同：evidence 里的补丁**永不应用**，这里的会真正打进
本地源码树用于构建算子扩展。

| 补丁 | 问题 |
|---|---|
| `OPSNN-1-cxx-std.patch` | JIT 构建器硬编码 `-std=c++17`，torch ≥ 2.9 的头文件要求 C++20，导致所有 JIT 算子构建失败 |
