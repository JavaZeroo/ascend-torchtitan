# shim 与 override：机制设计

> 修订 2026-09-02。取代 `docs/shim-override-architecture.html`（旧版只讲了"怎么写"，
> 没讲"什么时候不该写"和"怎么让它消失"）。
> 依据：`docs/PRINCIPLES.md` P0–P14、`docs/adr/ADR-001..007`，以及 2026-08 至 09
> 的实测经验——**本文里所有的规则，都是踩过对应的坑之后加上的**。

---

## 1. 目标

torchtitan 是为 CUDA 写的。要在昇腾 910B2 上原样跑它，必然有一批地方需要改。
本机制回答的是：**这些改动应该长什么样。**

### 1.1 功能目标

| # | 目标 | 验收 |
|---|---|---|
| F1 | 不 fork 上游 | `constraints/torchtitan.sha` 固定一个 commit，检出目录零改动（ADR-001/003） |
| F2 | 任何上游 flavor 都能跑，且能选择"原样跑"还是"叠加昇腾增量" | 每个 flavor 两个入口：裸名 / `_npu` 后缀 |
| F3 | 用户能挑单个模块替换，不改代码 | `--override.imports <module.function>` |
| F4 | 每处改动可审计：改了什么、为什么、谁的问题、什么时候能删 | 注册期强制，缺一项 CI 挂 |
| F5 | 改动能自动消失 | 特性探测，不是版本号比较 |

### 1.2 非功能目标（NFR）

| # | NFR | 怎么保证 |
|---|---|---|
| N1 | **可度量** | shim 数量、override 数量、矩阵红格数都是健康度指标，趋近于零 |
| N2 | **响亮失败** | 依赖缺失 → WARNING + provenance；绝不静默降级（P7 / ADR-004） |
| N3 | **可复现** | 任何 🟢 必须附命令与输出，且在 NIGHTLY 基线上跑过（P13） |
| N4 | **单一事实来源** | 版本在 `constraints/`，问题状态在 `STATUS.md`，别处只引 ID（P11） |
| N5 | **可迁移** | 机制与模型解耦，能整体搬到 TorchTitanTurbo |

### 1.3 非目标

- **不做"没有昇腾也能跑"的模式。** `torch` / `torch_npu` / `torchtitan` 硬导入，缺了就抛（P14 / ADR-007）。
- **不绕 torch_npu 的缺陷。** 归因 NPU 的问题只能修上游（P1 / P9），这是红线。
- **不追求 override 覆盖一切。** 上游没有 `Configurable` 节点的地方，去请上游抽一个，而不是替换父块（P6）。

---

## 2. 设计思路：一处改动的形态由三个问题决定

```mermaid
flowchart TD
    A["上游某处在昇腾上跑不了"] --> B{"torchtitan 已经<br/>有配置开关吗？"}
    B -->|有| C["用开关<br/>P0：一行配置不是负债"]
    B -->|没有| D{"根因在<br/>torch_npu 吗？"}
    D -->|是| E["不许绕<br/>P1/P9：修上游，矩阵标红"]
    D -->|否| F{"要改的是<br/>配置树上的节点，<br/>还是代码？"}
    F -->|"Configurable 节点"| G["override<br/>P6：只换已有节点"]
    F -->|"代码，没有节点"| H{"这是护栏，<br/>还是机制？"}
    H -->|"护栏（raise/assert）"| I["不许 shim 掉<br/>新规则 R1：见 5.4"]
    H -->|"机制"| J["shim<br/>P3/P4：包装优先，挂 issue"]

    style C fill:#d5f5d5
    style G fill:#d5f5d5
    style J fill:#fff3cd
    style E fill:#f8d7da
    style I fill:#f8d7da
```

**这张图的四个出口，代价递增**：配置 < override < shim < 修上游。
每往右一步都要有理由，而且理由要写进代码里，不是写进 commit message。

---

## 3. 原理

### 3.1 为什么是两套机制，而不是一套

torchtitan 的可扩展点分两种，它们的**生命周期不同**，所以不能用同一套机制：

```mermaid
graph LR
    subgraph CFG["配置树（运行时数据）"]
        direction TB
        C1["Trainer.Config"] --> C2["ModelSpec"]
        C2 --> C3["Decoder.Config"]
        C3 --> C4["GQAttention.Config"]
        C4 --> C5["FlexAttention.Config"]
    end
    subgraph CODE["模块代码（import 期常量）"]
        direction TB
        K1["FlexAttention._compiled_flex_attn<br/>ClassVar，import 时 torch.compile"]
        K2["_compiled_create_block_mask<br/>模块级，import 时 torch.compile"]
        K3["utils.device_type<br/>import 时冻结"]
    end
    CFG -.->|"能在构造后替换"| OV["override 机制"]
    CODE -.->|"构造前就定死了"| SH["shim 机制"]

    style OV fill:#d5f5d5
    style SH fill:#fff3cd
```

| | override | shim |
|---|---|---|
| 作用对象 | 配置树上的 `Configurable.Config` 节点 | 模块属性 / 类属性 |
| 时机 | 配置构造后、`build()` 前 | **第一次 `import torchtitan` 之前** |
| 用户可控 | ✅ `--override.imports` | ❌ 全局生效，只能用环境变量整条关掉 |
| 上游是否知情 | ✅ 上游自己的机制 | ❌ 猴补丁，上游随时可能挪走目标 |
| 负债性质 | 低——上游支持的扩展点 | 高——**每条都必须挂 issue，上游修好即删** |

### 3.2 分层

```mermaid
graph TD
    subgraph L4["工具层"]
        T1["doctor<br/>环境探针"]
        T2["matrix<br/>能力矩阵"]
        T3["release_check"]
    end
    subgraph L3["models/ 每个模型一个包"]
        M1["qwen3"]
        M2["qwen3_5"]
        M3["llama3"]
        M4["kimi_k3"]
    end
    subgraph L2["recipes/ 增量原语"]
        R1["deltas.py<br/>add_override / swap_override / flex_to_varlen"]
        R2["matrix.py<br/>npu_minimal / npu_fused"]
    end
    subgraph L1["kernels/ 昇腾算子"]
        K1["attention"]
        K2["gdn / kda"]
        K3["rms_norm / swiglu / rope"]
    end
    subgraph L0["compat/ shim 注册表"]
        S1["registry.py"]
        S2["shims/*.py"]
    end
    subgraph UP["上游（只读，SHA 锁定）"]
        U1["torchtitan"]
        U2["torch"]
        U3["torch_npu"]
    end

    M1 & M2 & M3 & M4 --> R1
    R1 --> R2
    R1 --> K1 & K2 & K3
    K1 & K2 & K3 -->|"@override"| U1
    S2 -->|"猴补丁"| U1
    S1 --> S2
    T2 --> R2
    T1 --> L0
    U1 --> U2 --> U3

    style L0 fill:#fff3cd
    style L1 fill:#d5f5d5
    style UP fill:#e9ecef
```

**依赖只能向下。** `kernels/` 不认识 `models/`；`recipes/deltas.py` 不认识具体模型；
模型的 `recipes.py` 逐条写出自己需要什么——**读那个函数就知道换了什么、没换什么**。

### 3.3 生命周期：一次训练启动

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant T as ascend_titan.train
    participant B as _bootstrap.setup
    participant NPU as torch_npu
    participant SH as compat.apply_all
    participant TT as torchtitan
    participant OV as config.apply_overrides
    participant TR as Trainer

    U->>T: python -m ascend_titan.train --config X_npu
    T->>B: setup()
    B->>NPU: import torch, import torch_npu
    Note over B,NPU: P14 硬导入，缺了就抛
    B->>B: 检查 device_type == "npu"
    Note over B: 不是 npu 就在这里抛，<br/>否则 torchtitan 会冻结成 cuda，<br/>几层之外报一个看不懂的错
    B->>SH: apply_all()
    SH->>SH: _discover() 导入 shims/*.py<br/>装饰器在 import 期校验
    SH->>TT: setattr(owner, attr, patched)
    Note over SH,TT: 必须早于第一次 import torchtitan
    SH-->>B: [Applied]
    B-->>T: SetupReport
    T->>TT: from torchtitan.train import main
    Note over TT: 此刻 device_type 才被冻结，已是 npu
    TT->>TT: 构造 Trainer.Config（含 override.imports）
    TT->>OV: apply_overrides(config.override, config)
    OV->>OV: resolve → collect claims → 冲突检查
    OV->>OV: 逐个 setattr 替换节点
    OV-->>TT: 返回替换日志行 FlexAttention.Config → AscendFusionAttention.Config
    TT->>TR: build() 用替换后的配置树
    TR-->>U: step: 1 loss: ...
```

**这张图里唯一不能动的是顺序**：`torchtitan/tools/utils.py` 在 import 期就用
`torch._utils._get_available_device_type()` 把 `device_type` 冻死。
`setup()` 晚一步，整个训练就跑在 `cuda` 分支上。
`_bootstrap` 因此会检测"torchtitan 是否已被导入"并给出警告。

---

## 4. 具体设计

### 4.1 shim 注册表

```mermaid
classDiagram
    class Shim {
        +str name
        +str target
        +str reason
        +str upstream
        +Kind kind
        +Callable fn
        +str why_not_wrap
        +module() str
        +attr() str
        +owner_path() list
        +owner(module) object
    }
    class Applied {
        +str name
        +str target
        +Kind kind
    }
    class Registry {
        -dict _REGISTRY
        -dict _APPLIED
        +shim(...) decorator
        +list_shims() list
        +apply_all(only) list
        +reset_for_tests()
    }
    Registry "1" o-- "*" Shim
    Registry "1" o-- "*" Applied
    Shim ..> Applied : 应用后产生
```

| 字段 | 含义 |
|---|---|
| `target` | `"module:attr"`，`attr` 可以是点分路径以触达类属性（`"pkg.mod:Class.attr"`） |
| `upstream` | 上游 issue/PR 的 URL，或 `draft:<path>#<anchor>` 指向 `docs/` 里已写好的正文 |
| `kind` | `wrap` / `replace` / `polyfill` |
| `fn(original)` | 收到当前属性，返回替换物；`wrap` 必须调用 `original` |
| `why_not_wrap` | `kind="replace"` 时必填 |

**注册期校验**（`shim()` 装饰器里，import 就跑，所以坏 shim 挂的是 CI 不是训练任务）：

```mermaid
flowchart TD
    A["@shim(...)"] --> B{"target 形如<br/>module:attr ?"}
    B -->|否| X1["ShimError"]
    B -->|是| C{"reason 非空?"}
    C -->|否| X2["ShimError"]
    C -->|是| D{"upstream 非空?"}
    D -->|否| X3["ShimError: P4<br/>先提 issue 再写 shim"]
    D -->|是| E{"upstream 是 http<br/>或 draft: ?"}
    E -->|否| X4["ShimError"]
    E -->|是| F{"kind == replace ?"}
    F -->|是| G{"why_not_wrap 非空?"}
    G -->|否| X5["ShimError: P3<br/>替换必须说明为什么不能包装"]
    G -->|是| H["注册"]
    F -->|否| H
    H --> I{"重名?"}
    I -->|是| X6["ShimError"]
    I -->|否| J["_REGISTRY[name] = Shim"]

    style X1 fill:#f8d7da
    style X2 fill:#f8d7da
    style X3 fill:#f8d7da
    style X4 fill:#f8d7da
    style X5 fill:#f8d7da
    style X6 fill:#f8d7da
    style J fill:#d5f5d5
```

**三种 kind，语义不同**：

```mermaid
flowchart LR
    subgraph W["wrap（默认，P3 首选）"]
        W1["original"] --> W2["patched = fn(original)"]
        W2 --> W3["patched 内部调用 original"]
        W3 --> W4["上游改了 original，<br/>我们自动继承"]
    end
    subgraph R["replace（须写 why_not_wrap）"]
        R1["original"] --> R2["patched = fn(original)"]
        R2 --> R3["patched 不调用 original"]
        R3 --> R4["上游的改进会被静默丢掉"]
    end
    subgraph P["polyfill（版本债）"]
        P1{"属性已存在?"} -->|是| P2["跳过，零改动"]
        P1 -->|否| P3["patched = fn(None)"]
    end

    style W4 fill:#d5f5d5
    style R4 fill:#f8d7da
    style P2 fill:#d5f5d5
```

`polyfill` 的价值在于**它会自己消失**：上游长出这个属性，`apply_all` 直接跳过并记一条 INFO，
不需要任何代码改动。这是"消失条件"最理想的形态。

### 4.2 shim 的生命周期

```mermaid
stateDiagram-v2
    [*] --> 提出: 发现上游代码在昇腾上跑不了
    提出 --> 拒绝: 有配置开关（P0）
    提出 --> 拒绝: 根因在 torch_npu（P1/P9）
    提出 --> 拒绝: 目标是护栏而非机制（R1）
    提出 --> 注册: 写出 reason + upstream + 消失条件
    注册 --> 生效: apply_all() 且特性探测为真
    注册 --> 让路: 特性探测为假（换了芯片/装了包/上游修了）
    生效 --> 让路: 环境变化
    让路 --> 生效: 环境变回去
    让路 --> 删除: 上游合入，探测恒为假
    生效 --> 删除: 上游合入
    删除 --> [*]
    拒绝 --> [*]

    note right of 让路
        关键：让路是运行时行为，
        不是版本号判断。
        换到 Ascend950 自动生效，
        不需要改一行代码。
    end note
```

### 4.3 override：claim → 冲突检查 → 替换

```mermaid
flowchart TD
    A["override.imports<br/>目标字符串，或 目标+kwargs 二元组"] --> B["_resolve_target<br/>import 模块，触发 @override 注册"]
    B --> C["_resolve_active<br/>取出 Override 对象"]
    C --> D["_collect_claims<br/>遍历配置树，按 target_cls + fqns glob 匹配"]
    D --> E["_check_node_conflicts"]
    E --> F{"同一节点<br/>被两条 claim?"}
    F -->|是| X1["ValueError<br/>收窄 fqns"]
    F -->|否| G{"一条 claim 是<br/>另一条的祖先?"}
    G -->|是| X2["ValueError<br/>嵌套 override 顺序相关"]
    G -->|否| H["逐个 factory(cfg, **kwargs)"]
    H --> I["setattr(parent, attr, new_cfg)"]
    I --> J["返回 [Override] 日志行"]

    style X1 fill:#f8d7da
    style X2 fill:#f8d7da
    style J fill:#d5f5d5
```

**先收集再改（collect-then-mutate）**：如果边遍历边替换，冲突就变成顺序相关的——
先跑的那条赢，而"先跑"取决于 `imports` 列表的顺序。收集完再检查，冲突是确定性的错误。

**两种冲突形态**：

```mermaid
graph TD
    subgraph A["形态一：同一节点"]
        A1["layers.0.attention.inner_attention"]
        A2["override X"] -->|claim| A1
        A3["override Y"] -->|claim| A1
        A1 --- A4["❌ 谁赢取决于顺序"]
    end
    subgraph B["形态二：祖先 / 后代"]
        B1["layers.0.attention"] --> B2["layers.0.attention.inner_attention"]
        B3["override X"] -->|claim| B1
        B4["override Y"] -->|claim| B2
        B2 --- B5["❌ X 换掉整块，Y 改的节点已经不在树上了"]
    end

    style A4 fill:#f8d7da
    style B5 fill:#f8d7da
```

形态二是真实踩过的坑：上游 `fused_mla` 认领 `layers.N.attention`（整个注意力块），
它同时是 inner attention 与 RoPE 两个节点的祖先，所以我们那两条 override **都**得跳过，
不是只跳 RoPE。

**嵌套配置怎么换：`derive`**

当一个节点需要连同它的子节点一起换（KDA 就是：`InnerKDA.Config` 持有
`kernel: KDAKernel.Config`），不能写两条 override（会触发形态二），
而是**一条 override 在父节点上，用 `derive` 就地构造子节点**：

```mermaid
graph LR
    subgraph BEFORE["替换前"]
        P1["InnerKDA.Config"] --> C1["KDAKernel.Config<br/>要 CUDA + Blackwell"]
    end
    subgraph AFTER["替换后（一条 override）"]
        P2["AscendKDA.Config"] --> C2["AscendKDAKernel.Config<br/>derive(cfg.kernel, ...)"]
    end
    BEFORE -->|"@override(target=InnerKDA.Config)"| AFTER
```

`derive` 的价值是**版本鲁棒**：目标 Config 以后长出新字段时，同名字段自动从 source 复制，
而不是静默退回默认值。工厂函数只写它真正要改的那几项。

### 4.4 门控（gate）：本项目最贵的一课

一条 shim / 一个增量什么时候生效，由**特性探测**决定。这里有三个独立的坑，我们三个都踩过。

#### 坑一：探果不探因

```mermaid
flowchart TD
    A["需要判断：<br/>编译版 flex 能不能用？"] --> B["❌ 探 triton.runtime.driver.active<br/>（有没有 triton 后端）"]
    A --> C["✅ 探 torch_npu._inductor.config<br/>.inductor_indirect_memory_mode<br/>（能不能 lower 间接寻址）"]
    B --> D["装上 Triton-Ascend 后<br/>探测返回 True"]
    D --> E["flex_to_varlen 停止转换"]
    E --> F["llama3 一次要 20 GiB → OOM"]
    C --> G["只在 Ascend950 上非 None"]
    G --> H["910B2 恒为假 → 继续转换<br/>Ascend950 自动让路"]

    style B fill:#f8d7da
    style F fill:#f8d7da
    style C fill:#d5f5d5
    style H fill:#d5f5d5
```

**规则**：探测必须指向**根因本身**。"有没有 triton 后端"和"能不能 lower document mask"
是两件事；把前者当后者，装个包就会把结论翻转过来。
修对这个探测之后，`3d_compile` 从红变绿——**同一个修正同时解决了假绿和假红**。

#### 坑二：判断时机

```mermaid
sequenceDiagram
    participant S as setup()
    participant SH as shim 应用
    participant D as set_determinism()
    participant R as 训练循环

    Note over S,R: ❌ 一次性判断
    S->>SH: apply_all()
    SH->>SH: 判断 deterministic 后决定是否换 eager
    Note over SH: 此刻 deterministic 还是 False<br/>永远读到 False
    S->>D: torchtitan 设置 deterministic=True
    D->>R: 训练开始（shim 已经决定不生效）

    Note over S,R: ✅ 每次调用判断
    S->>SH: apply_all()
    SH->>SH: 装一个 dispatcher
    S->>D: deterministic=True
    R->>SH: 调用 build_block_mask()
    SH->>SH: 此刻才判断 → 走 eager
```

**规则**：shim 在 `setup()` 期应用，比 torchtitan 的大部分运行时状态都早。
凡是依赖运行时状态的门控，替换物必须是**分发器**，逐次调用判断。

#### 坑三：门开得太窄，等于没有

```mermaid
flowchart TD
    A["掩码构建编不出来"] --> B["观察到的现场：<br/>kimi_k3 在确定性模式下失败"]
    B --> C["❌ 门 = 确定性模式"]
    C --> D["stock qwen35_debugmodel<br/>在普通运行里死掉"]
    D --> E["结论被记成<br/>'910B2 跑不了 stock flex'"]
    E --> F["把我们的问题<br/>归给了硬件"]
    B --> G["✅ 门 = 芯片能不能 lower 间接寻址<br/>（或 确定性模式）"]
    G --> H["stock qwen35_debugmodel<br/>2 步跑通 12.72494 → 12.56159"]

    style C fill:#f8d7da
    style F fill:#f8d7da
    style G fill:#d5f5d5
    style H fill:#d5f5d5
```

**规则**：门控条件要从**根因**推出来，不能从**观察到的现场**归纳。
kimi_k3 的掩码恰好不落在间接寻址那个模式里，所以它只在确定性模式下暴露；
拿单个模型的现象当芯片结论，是这次归因错误的全部原因。

---

## 5. 案例

### 5.1 注意力：一条链路，两个入口

```mermaid
flowchart LR
    subgraph UPSTREAM["上游只有两种 inner attention"]
        F["FlexAttention"]
        V["VarlenAttention"]
    end
    subgraph WHY["为什么都不能直接用"]
        F --> F1["编译版：910B2 lower 不了 document mask"]
        F --> F2["eager 版：实体化 O(T²) 分数矩阵 → OOM"]
        V --> V1["要 aten::_flash_attention_forward<br/>torch_npu 没有（NPU-1）"]
    end
    subgraph OURS["两条 override，不同 target"]
        O1["npu_fusion_attention_from_flex<br/>target=FlexAttention.Config"]
        O2["npu_fusion_attention<br/>target=VarlenAttention.Config"]
    end
    F1 --> O1
    V1 --> O2
    O1 & O2 --> A["AscendFusionAttention<br/>torch_npu.npu_fusion_attention<br/>TND + sparse_mode=3"]

    style A fill:#d5f5d5
```

两条 override target 不同的 Config 类，**可以同时激活**——上游不同 flavor 的默认节点不一样。

### 5.2 CP：为什么这条 override 必须**不**生效

```mermaid
flowchart TD
    A["CP 开启"] --> B{"注意力节点是什么？"}
    B -->|"FlexAttention"| C["torch 有 CP 实现<br/>flex_cp_allgather：all-gather K/V"]
    B -->|"SDPA"| D["torch 有 CP 实现<br/>ring attention"]
    B -->|"VarlenAttention<br/>或我们的融合算子"| E["torch 没有钩子"]
    E --> F["上游 decoder.py 抛<br/>NotImplementedError"]
    C --> G["✅ 能跑（eager flex，显存换正确性）"]
    D --> G
    F --> H["❌ 硬失败"]

    style G fill:#d5f5d5
    style H fill:#f8d7da
```

所以 `flex_to_varlen` 在 `context_parallel_degree > 1` 时**直接返回**：
非 CP 场景下救命的转换，在 CP 下只会把能跑的变成跑不了的。
**一个增量的适用条件，本身就是设计的一部分。**

### 5.3 GDN：一条 override 覆盖整棵子树

见 4.3 的 `derive` 图。要点：父子节点不能各写一条 override（形态二冲突），
父节点上一条 + `derive` 换子节点。

### 5.4 新规则 R1：护栏不许 shim 掉

这是本次评审新增的规则，起因是一个**差点做错的决定**。

```mermaid
flowchart TD
    A["上游 decoder.py:186<br/>raise NotImplementedError<br/>'CP 不支持 varlen'"] --> B{"这是什么？"}
    B --> C["护栏：<br/>它保护的机制在 torch 里，<br/>而 torch 只给 flex/SDPA 实现了 CP"]
    C --> D["shim 掉这个 raise 会怎样？"]
    D --> E["每张卡只在自己那段序列上算注意力"]
    E --> F["loss 静默算错，不报任何错"]
    F --> G["❌ 最坏的失败模式<br/>违反 P7"]
    C --> H["✅ 正确做法：<br/>要么不触发它（CP 下不转 flex），<br/>要么实现它保护的机制"]

    style G fill:#f8d7da
    style H fill:#d5f5d5
```

**R1：一处 `raise` / `assert` 是护栏还是机制，决定它能不能被 shim。**
护栏背后如果有未实现的机制，绕过它得到的不是"能跑了"，是"错得没有声音"。
判断方法：**问这个 raise 保护的东西在哪里实现**。找不到实现，就是护栏。

---

## 6. 决策记录

已有 ADR 见 `docs/adr/`。本文新增两条待立项决策：

### ADR-008（提案）：门控探测根因，不探代理信号

**Context.** `_flex_attention_is_usable()` 最初探 `triton.runtime.driver.active`。
装上 Triton-Ascend 后它返回 True，flex→varlen 转换停止，llama3 一次申请 20 GiB 而 OOM。

**Decision.** 特性探测必须指向根因本身可观测的开关。
本例是 `torch_npu._inductor.config.inductor_indirect_memory_mode`。

**Alternatives.**
- 版本号比较——被 P12 排除：版本号不描述硬件能力。
- 试跑一次再回退——代价不可控，且掩盖归因。

**Consequences.** 正面：换硬件自动让路，无需改代码；同一次修正让 `3d_compile` 从红转绿。
负面：探测点是上游私有属性，上游改名会失效——因此探测必须写在一个函数里，集中失效。

### ADR-009（提案）：护栏不可 shim

**Context.** 上游 CP 的 `NotImplementedError` 看起来像一条"限制"，删掉它就能跑。
实际它保护的机制（CP 下的注意力集合通信）在 torch 里只对 flex 与 SDPA 存在。

**Decision.** shim 不得移除 `raise` / `assert`，除非同时提供它保护的机制。

**Alternatives.** 删掉后跑通再说——被 P7 排除：结果是静默的错误 loss。

**Consequences.** 正面：杜绝一类无声的数值错误。
负面：某些能力要等我们自己实现机制（如让 CP 走融合算子），周期更长。

---

## 7. 风险与对策

```mermaid
graph TD
    R1["风险：shim 目标被上游挪走"] --> M1["对策：apply_all 找不到目标就抛，<br/>错误信息里带 upstream 链接"]
    R2["风险：shim 越积越多"] --> M2["对策：数量是健康度指标；<br/>每条挂 issue，上游合入即删"]
    R3["风险：override 与上游 override 抢节点"] --> M3["对策：collect-then-mutate + 祖先检查；<br/>npu_minimal 检测到上游 override 就跳过"]
    R4["风险：门控写错导致假绿/假红"] --> M4["对策：ADR-008；<br/>矩阵三态 + 归因规则即数据"]
    R5["风险：静默降级污染性能数据"] --> M5["对策：P7 响亮降级 + provenance 表；<br/>无 provenance 的 benchmark 不收"]
    R6["风险：结论基于单个模型的现象"] --> M6["对策：stock 对照组<br/>（__stock 模式，零 override）"]

    style M1 fill:#d5f5d5
    style M2 fill:#d5f5d5
    style M3 fill:#d5f5d5
    style M4 fill:#d5f5d5
    style M5 fill:#d5f5d5
    style M6 fill:#d5f5d5
```

**R6 的对策值得单独说**：能力矩阵的每个格子都可以用 `__stock` 后缀跑一遍零 override 的对照。
这次把 CP 归因错误、TA-1 影响面、qwen3.5 的 MoE 能力三个结论纠正回来，靠的都是它。
**没有对照组，就分不清"上游的问题"和"我们的问题"。**

---

## 8. 度量

| 指标 | 现在 | 目标 | 来源 |
|---|--:|---|---|
| shim 数量 | 3（2 个文件） | → 0 | `list_shims()` |
| 其中 `kind=replace` | 2 | → 0（能包装就包装） | 同上 |
| 无消失条件的 shim | 0 | 0 | 人工评审 |
| 矩阵红格 | 13 / 61 | 只剩"上游按设计 CUDA-only" | `docs/matrix/` |
| 红格中归因"我们" | 0 | 0 | 同上 |
| 红格中归因 torch_npu | 0 | 0 | 同上 |

```bash
python -m ascend_titan.tools.doctor          # 打印已注册 shim
python -m ascend_titan.models.registry       # 打印模型支持状态
python -m ascend_titan.tools.matrix --list   # 打印能力矩阵用例
```

---

## 9. 迁移到 TorchTitanTurbo

机制与模型解耦，可以整体搬。建议顺序：

```mermaid
graph LR
    P0["RFC：版本策略<br/>SHA 锁 + nightly-first"] --> P1["PR1：shim 注册表<br/>compat/ 整体搬"]
    P1 --> P2["PR2：显式 setup()<br/>顺序约束 + device_type 检查"]
    P2 --> P3["PR3：override 卫生<br/>deltas 原语 + 冲突规避"]
    P3 --> P4["PR4：能力矩阵<br/>三态 + 归因即数据 + stock 对照"]

    style P0 fill:#e9ecef
```

先搬 `compat/`（它零依赖、自带校验、价值最直观），再搬 `setup()` 的顺序约束——
后者是**最容易被忽略、代价最大**的一条：晚一步 import，整个训练跑在错误的 device 分支上。
