# <Model> on Ascend

> 骨架：`cp -r ascend_titan/models/_template ascend_titan/models/<model>`，
> 把 `recipes.py.txt` / `__init__.py.txt` 去掉 `.txt` 后缀，再全局替换 `<Model>` / `<model>` / `<flavor>`。
> 模板文件用 `.txt` 后缀，因此不会被 import、lint 或收集为测试。

**状态 🟢/🟡/🔴/⚪** · 一句话结论。

| | |
|---|---|
| 上游模型包 | `torchtitan.models.<model>` |
| 我们的 recipe | `ascend_titan/models/<model>/recipes.py` |
| golden | `tests/assets/losses/npu/<fn>__torch<v>_npu<v>.txt`（没有就写"无"） |
| 最近验证 | torch <v> + torch_npu <v>，CANN <v>，<日期> |
| 阻塞 | 🔴 时必填：报错原文 + 归因标签 |

## 1. 五分钟跑起来
命令 + **真实的**预期输出（loss 数字来自实跑，不要编）。

## 2. 有哪些 recipe
表格：函数 / 卡数 / 说明。probes.py 里的函数单独一张表，并标注"只用于测量"。

## 3. 增量逐条解释
表格：# / 增量 / 为什么 / 什么时候能删。**"什么时候能删"是必填项**——没有到期日的增量会永远活下去（P12）。

## 4. 真实尺寸
上游 registry 里还有哪些 flavor，各自什么状态（未跑过就写 ⚪，不要写"应该可以"）。
跑法：走矩阵 runner 跑上游配置，别新建配置。

## 5. 待办
可以删掉的增量、待录的 golden、待解的阻塞。
