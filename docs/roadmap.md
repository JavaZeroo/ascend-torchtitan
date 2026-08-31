# 路线图

现状看 `ascend_titan/models/registry.py`（每个模型的 R1–R8）与 `docs/capability-matrix.md`
（每个特性的三态与归因）。这里只写**接下来做什么**，按门槛分期——每一期的解锁条件是进入
下一期的前提，顺序不能调换。

## 一 · 把线性注意力做实

解锁条件：**已达成**——`fla-npu` 已在 NIGHTLY 上构建安装并通过对拍。

`flash-linear-attention-npu`（`../flash-linear-attention-npu`，AscendC 实现）现已可用：
`build.sh --pkg --soc=ascend910b --vendor_name=fla_npu` 出 `.run`，再 `torch_custom/fla_npu/build.sh`
编 wheel 并把 OPP 叠进去（顺序不能反）。它的 pip 步骤带 `--no-deps`，实测没有动 torch 三元组。

`tests/repro/probe_solve_tri_crosscheck.py` 是那个**独立于 attn_gym 的第二实现**互证——
qwen3.5 发散事件的直接对策。两边都对 float64 参考解收敛，bf16 下我们 4.07e-3 / 它 1.42e-3
（C=64），彼此之差就是 bf16 噪声；这独立佐证了当初把 Neumann doubling 换成分块前代的改写
是对的。速度上它明显更好：C=64 时 0.467 ms vs 我们 1.356 ms，且 C 翻倍它几乎不变而我们线性翻倍。

接下来做 R5（GDN 现在是纯 torch 的 chunk 递推，245 tps vs Qwen3 的 10,186 tps）时，三条实测约束：

- **导入时机是硬约束**。`fla_npu` 必须在第一个 NPU 算子执行之前 import——它在 `__init__` 里设
  `ASCEND_CUSTOM_OPP_PATH`，而 CANN 的算子注册表一旦初始化就不再重读。晚导入会以
  `aclnnStatus=161001`（NULLPTR，查不到算子）失败。接进来时导入点应当在 `_bootstrap.setup()`。
- **`npu_solve_tri` 不是我们那条路径的直接替换**。它在 `solve_tri_def.cpp` 里把输入 dtype 写死成
  `{DT_FLOAT16, DT_BF16}`，而我们的 GDN 契约是 FP32 递推（fp32 下我们 rel≈1e-7，bf16 下退到 4e-3）。
  换过去等于拿数值精度换速度，而这个算子恰好就是当初咬人的地方——要换必须先有 R4 级别的取证。
- **他们的整网示例跑不起来**，`examples/flash_gated_delta_rule.py` 的 `l2norm` / `chunk_local_cumsum`
  等走 Triton-Ascend，本机没装，报 `0 active drivers`（和 flex attention 那次同一个缺口）。
  可用的是 AscendC 那一批直接调用，非 AscendC 的部分继续用我们自己的 torch 实现。
- **补齐 Qwen3.5 的 R2 / R4 / R6**。TP / PP / EP 未测；DCP 续训要先有一个冻结视觉塔的
  机制（纯文本 recipe 让视觉塔没有优化器状态，上游只有 LoRA 自己做冻结）；R6 被 R5 卡住
  （一步 67 秒，500 步要 9 小时）。后两条依赖上一件事先落地。

## 二 · 把取证成本压下来

解锁条件：nightly CI 跑在真实 runner 上。

- **真实 runner 上的 nightly**。矩阵扫描与 `release_check` 目前都是手动触发。自动化之后
  才能在 torch_npu 重建导致回归时当天发现。
- **找到有效的廉价代理**。debugmodel golden 抓不到真实尺寸的数值问题——0.8B 发散时它全绿，
  而修掉那个发散的改写让它逐位不变。需要介于两者之间的东西：固定输入下的逐层激活范数快照，
  或算子级的极端输入压测（已有雏形）。
- **对上游 eager 的对拍覆盖到每个 override**。防的是上游改语义而不改名字这种静默漂移。
  改名字会在 import 时炸，改语义不会。

## 三 · 扩模型覆盖

解锁条件：前两期就位，单个模型的验证成本可控。

- **从 4 个专属 recipe 扩到上游主力集**。DeepSeek-V3、GPT-OSS、Kimi K2.7 目前只有矩阵覆盖。
  加 recipe 本身是 trivial 的，成本全在取证——必须等第二期把成本压下来，否则只是把一堆 ⚪
  变成另一堆 ⚪。
- **Qwen3 的其它尺寸**。1.7B / 14B / 32B / 30B-A3B。14B 在 8×910B2 上已确认装不下
  （`FullAC` + 1×4096 微批仍 OOM），更大的尺寸要先解决显存配平。
- **MoE / EP 的深度覆盖**。矩阵里 `fsdp+ep` 已绿，但没有一个 MoE 模型走完 R1–R8。

## 四 · 硬件相关

- **上下文并行重测**。此前判定"910B2 上不可能"的前提（flex 编不出来）已被推翻，需要按实测
  重走一遍。NIGHTLY 扫描里 CP 的 12 个红格归因是 `CANN`，前提没了不等于它就能跑。
- **FP8**。910B2 上张量能分配，但所有转换报 `aclnnInplaceCopy 561103`、`_scaled_mm` 明确
  要求 Ascend950。按"不加投机性代码"，post-converter 树上的 FP8 override 至今没写——没有
  硬件就无从验证。
- **OURS-13**：自定义算子缺 GE converter，整模型进不了 torchair 图。

## 贯穿始终

`kernels/` 应该越来越薄。今天它是仓库里代码量最大的部分，但每一个昇腾侧融合算子成熟，
就应该有一段自研数值代码退化成适配器。如果这一层还在长大，说明昇腾的算子生态没有跟上——
那是这个项目真正的长期风险。

## 明确推迟的决策

替换型 shim 的源码指纹（只在真出现替换型 shim 时做）、`AscendTrainer` 子类（尚无必要场景）、
vendor 无关中间层（不做）。
