---
name: 融合算子 / override 需求
about: 用昇腾算子替换一个上游 Configurable 节点
labels: kernels
---
**上游节点**（`torchtitan/.../file.py:line`，`X.Config`）：

**算子**（包 + op，例如 ops-nn `situ_glu`）：

**该计算有自己的 `Configurable` 节点吗？**（没有的话就是一个上游 ask——P6）

**数值参考**（用来对齐的上游 eager 路径）：
