---
name: Fused kernel / override request
about: Replace an upstream Configurable node with an Ascend kernel
labels: kernels
---
**Upstream node** (`torchtitan/.../file.py:line`, `X.Config`):

**Kernel** (package + op, e.g. ops-nn `situ_glu`):

**Does the computation have its own `Configurable` node?** (if not, this is an upstream ask — P6)

**Numerics reference** (upstream eager path to align against):
