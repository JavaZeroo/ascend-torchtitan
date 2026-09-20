# shim 与 override：机制设计

> 面向**尚未实现该机制**的 torchtitan 树外扩展。上游按 commit SHA 固定，扩展住在树外。
> 渲染版（含全部图）：见 `docs/README.md` 里的 artifact 链接。

torchtitan 是为 CUDA 写的。要在别的加速器上原样跑它，必然有一批地方需要改。
这份文档不讲改哪些地方，只回答一个问题：**这些改动应该长什么样，才不会变成一个 fork。**

四个出口，代价递增。全文按这四类展开：

| 出口 | 什么时候走 |
|---|---|
| **配置** | 上游已有开关。一行配置不是负债。 |
| **override** | 要改的是配置树上已有的 `Configurable` 节点。上游自己的扩展点。 |
| **shim** | 要改的是 import 期就定死的代码。受治理的猴补丁，挂 issue，有到期日。 |
| **禁止** | 设备后端的缺陷，以及护栏。只能修上游。 |

每往右一步都要有理由，而且理由写进代码里，不是写进 commit message。

---

## 1. 设计思路

一处改动的形态，由三个问题依次决定。

```mermaid
flowchart TD
    A["上游某处在这个设备上跑不了"] --> B{"上游已经<br/>有配置开关吗？"}
    B -->|有| C["用开关<br/>一行配置不是负债"]
    B -->|没有| D{"根因在<br/>设备后端吗？"}
    D -->|是| E["不许绕<br/>修上游，测量里标红"]
    D -->|否| F{"要改的东西，在配置里<br/>有没有对应的 .Config ？"}
    F -->|"配置里有<br/>对应的 .Config"| G["override<br/>换掉那个 .Config 实例"]
    F -->|"只是代码，<br/>配置里没有对应物"| H{"要改的那一处，是一句拒绝，<br/>还是一段实现？"}
    H -->|"一句拒绝<br/>raise / assert"| I["不许 shim 掉<br/>规则 R1"]
    H -->|"一段实现"| J["shim<br/>包装优先，挂 issue"]
```

### 三个词的意思

| 词 | 指什么 | 怎么认出来 |
|---|---|---|
| **节点** | 配置树上的一个 `.Config` 实例。上游的配置是一棵 dataclass 树，每个可替换组件（注意力、loss、优化器、dataloader）在树上都有自己的 `.Config`。一个节点有地址，形如 `layers.0.attention.inner_attention`。 | 能用 FQN 指到它，就是节点。override 做的事就是把某个地址上的实例换成另一个类的实例。 |
| **一段实现**（机制） | 真正干活的代码：一个函数、一个 `torch.compile` 的产物、一个 import 期算好的常量。 | 它有输入有输出。换掉它，行为改变，语义仍然成立。 |
| **一句拒绝**（护栏） | 一句 `raise` / `assert`，用来挡住一个上游没有实现的组合。 | 问一句：**它挡住的那件事，实现在哪里？** 找不到实现，就是护栏——它不是限制，它是在替你避免一个错误的结果。 |

这三者的区别决定了改动的形态，所以它是这张图上唯一需要判断的地方；其余分支都是查一下有没有开关、归因在谁。

---

## 2. 原理

### 2.1 为什么是两套机制

上游的可扩展点分两种，**生命周期不同**，所以不能用同一套机制。

```mermaid
graph LR
    subgraph CFG["配置树 · 运行时数据"]
        direction TB
        C1["Trainer.Config"] --> C2["ModelSpec"]
        C2 --> C3["Decoder.Config"]
        C3 --> C4["GQAttention.Config"]
        C4 --> C5["FlexAttention.Config"]
    end
    subgraph CODE["模块代码 · import 期常量"]
        direction TB
        K1["某模块的 ClassVar<br/>import 时就 torch.compile 掉了"]
        K2["模块级常量<br/>import 时算好，之后不再变"]
        K3["device_type<br/>import 时冻结"]
    end
    CFG -.->|"能在构造后替换"| OV["override 机制"]
    CODE -.->|"构造前就定死了"| SH["shim 机制"]
```

| | override | shim |
|---|---|---|
| 作用对象 | 配置树上的 `Configurable.Config` 节点 | 模块属性 / 类属性 |
| 时机 | 配置构造后、`build()` 前 | **第一次 import 上游之前** |
| 用户可控 | ✅ `--override.imports` | ❌ 全局生效，只能整条关掉 |
| 上游是否知情 | ✅ 上游自己的机制 | ❌ 猴补丁，目标随时可能被挪走 |
| 负债性质 | 低——上游支持的扩展点 | 高——每条必须挂 issue，上游修好即删 |

### 2.2 分层

依赖只能向下。

```mermaid
graph TD
    subgraph L4["L4 工具"]
        T1["环境探针"]
        T2["能力矩阵"]
        T3["release 检查"]
    end
    subgraph L3M["L3 models 每个模型一个包"]
        M1["模型 A"]
        M2["模型 B"]
        M3["模型 C"]
        M4["..."]
    end
    subgraph L3R["L3 recipes 增量原语"]
        R1["deltas.py<br/>add_override / swap_override / 节点转换"]
        R2["matrix.py<br/>通用变换 minimal / fused"]
    end
    subgraph L1["L1 kernels 设备算子"]
        K1["attention"]
        K2["linear attention"]
        K3["norm / mlp / rope"]
    end
    subgraph L0["L0 compat shim 注册表"]
        S1["registry.py"]
        S2["shims/*.py"]
    end
    subgraph UP["上游 · 只读 · SHA 锁定"]
        U1["torchtitan"]
        U2["torch"]
        U3["设备后端"]
    end

    M1 --> R1
    M2 --> R1
    M3 --> R1
    M4 --> R1
    R1 --> R2
    R1 --> K1
    R1 --> K2
    R1 --> K3
    K1 -->|"@override"| U1
    K2 -->|"@override"| U1
    K3 -->|"@override"| U1
    S1 --> S2
    S2 -->|"猴补丁"| U1
    T2 --> R2
    T1 --> S1
    U1 --> U2
    U2 --> U3
```

`kernels/` 不认识 `models/`；`deltas.py` 不认识具体模型；模型包逐条写出自己需要什么——
**读那个函数就知道换了什么、没换什么。**

### 2.3 一次训练启动

这张图里唯一不能动的是顺序。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant T as 扩展包.train
    participant B as bootstrap.setup
    participant NPU as 设备后端
    participant SH as compat.apply_all
    participant TT as torchtitan
    participant OV as apply_overrides
    participant TR as Trainer

    U->>T: python -m 扩展包.train --config X_加速
    T->>B: setup()
    B->>NPU: import torch，再 import 设备后端
    Note over B,NPU: 基础依赖硬导入，缺了就抛
    B->>B: 检查 device_type 已是目标设备
    Note over B: 不对就在这里抛。否则 torchtitan 冻结成 cuda，<br/>几层之外报一个看不懂的错
    B->>SH: apply_all()
    SH->>SH: 导入 shims，装饰器在 import 期校验
    SH->>TT: setattr(owner, attr, patched)
    Note over SH,TT: 必须早于第一次 import 上游
    SH-->>B: [Applied]
    B-->>T: SetupReport
    T->>TT: from torchtitan.train import main
    Note over TT: 此刻 device_type 才被冻结，已经是对的
    TT->>TT: 构造 Trainer.Config，含 override.imports
    TT->>OV: apply_overrides(config.override, config)
    OV->>OV: 收集 claim → 冲突检查 → 替换
    OV-->>TT: 返回替换日志 原 Config → 替换后 Config
    TT->>TR: build() 用替换后的配置树
    TR-->>U: step 1 loss ...
```

上游在 import 期就用 `torch._utils._get_available_device_type()` 把 `device_type` 冻死。
`setup()` 晚一步，整个训练跑在 `cuda` 分支上。所以 bootstrap 要检测「上游是否已被导入」并告警。

---

## 3. 具体设计

### 3.1 shim 注册表

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
    }
    Registry "1" o-- "*" Shim
    Registry "1" o-- "*" Applied
    Shim ..> Applied : 应用后产生
```

### 3.2 target：要打的那个位置怎么写

`target` 是一个字符串，冒号左边是模块，右边是模块里的属性名；右边可以再用点分下去，
以触达类属性。一条 shim 因此只需要一行就能定位到任意一个可 `setattr` 的位置。

| 写法 | 打到哪里 |
|---|---|
| `"pkg.mod:func"` | 模块级函数或变量 |
| `"pkg.mod:Cls.attr"` | 类属性 / ClassVar——先取到 `Cls`，再对它 `setattr` |

注册表把 `target` 拆成三部分：`module`（冒号左边）、`owner_path`（中间那些点分名，用来一路
`getattr` 找到真正持有属性的对象）、`attr`（最后一段，`setattr` 的那个名字）。
`owner_path` 为空就打在模块上，否则打在类上。

### 3.3 注册期校验

校验跑在 `@shim` 装饰器里，也就是 import 期——所以一条写坏的 shim 挂的是 CI，不是半夜的训练任务。

```mermaid
flowchart TD
    A["@shim(...)"] --> B{"target 里有冒号吗？<br/>左边模块，右边属性"}
    B -->|否| X1["ShimError"]
    B -->|是| C{"reason 非空?"}
    C -->|否| X2["ShimError"]
    C -->|是| D{"upstream 非空?"}
    D -->|否| X3["ShimError<br/>先提 issue 再写 shim"]
    D -->|是| E{"upstream 是 http 或 draft: ?"}
    E -->|否| X4["ShimError"]
    E -->|是| F{"kind == replace ?"}
    F -->|是| G{"why_not_wrap 非空?"}
    G -->|否| X5["ShimError<br/>替换必须说明为什么不能包装"]
    G -->|是| H{"重名?"}
    F -->|否| H
    H -->|是| X6["ShimError"]
    H -->|否| J["注册进 _REGISTRY"]
```

### 3.4 三种 kind

```mermaid
flowchart LR
    subgraph W["wrap · 默认，优先选它"]
        W1["original"] --> W2["patched = fn(original)"]
        W2 --> W3["patched 内部调用 original"]
        W3 --> W4["上游改了 original，我们自动继承"]
    end
    subgraph R["replace · 须写 why_not_wrap"]
        R1["original"] --> R2["patched = fn(original)"]
        R2 --> R3["patched 不调用 original"]
        R3 --> R4["上游的改进会被静默丢掉"]
    end
    subgraph P["polyfill · 版本债"]
        P1{"属性已存在?"} -->|是| P2["跳过，零改动"]
        P1 -->|否| P3["patched = fn(None)"]
    end
```

| kind | 什么时候用 | 它干什么 | 解决什么问题 | 代价 |
|---|---|---|---|---|
| `wrap`（默认） | 上游那个函数本身是对的，只是**少做一步**你的设备需要的动作 | 收到 `original`，返回新函数；新函数内部调用 `original`，再在前后加行为 | 「上游没有为这个设备留分支」——你补上分支，主体逻辑还是上游的 | 几乎没有。上游以后改了 `original`，你自动继承 |
| `replace` | 上游那个对象**本身就是问题**，包一层没有意义 | 返回一个完全不同的实现，不调用 `original` | 典型是「上游在 import 期无条件 `torch.compile` 了某个函数」——包装它照样会进编译器，只有换掉才有用 | 高。上游对原实现的改进会被**静默丢掉**，所以注册时强制填 `why_not_wrap` |
| `polyfill` | 当前版本的上游**还没有**一个你依赖的 API | 属性不存在时补上；**已存在则整条跳过** | 版本差——你按 nightly 开发，用户可能在旧版上；作为 plugin 需要有「上游缺 API 就补」的能力 | 版本债。上游长出该属性后自动失效，然后应当删除 |

三者里只有 `polyfill` **会自己消失**：目标属性一旦存在，`apply_all` 记一条 INFO 就跳过。
`wrap` 与 `replace` 的消失要靠特性探测或上游合入后人工删除。

> **选型：先问「能不能包装」，不能再问「为什么不能」。**
> 把 `why_not_wrap` 做成注册期必填，是为了让「换掉上游实现」这个决定必须被写下来一次。
> 一条写不出理由的 `replace`，通常说明它其实应该是 `wrap`，或者根本不该存在。

### 3.5 shim 的生命周期

```mermaid
stateDiagram-v2
    [*] --> 提出: 发现上游代码在这个设备上跑不了
    提出 --> 拒绝: 上游已有配置开关
    提出 --> 拒绝: 根因在设备后端，只能修上游
    提出 --> 拒绝: 目标是护栏而非实现
    提出 --> 注册: 写出 reason + upstream + 消失条件
    注册 --> 生效: apply_all 且特性探测为真
    注册 --> 让路: 特性探测为假
    生效 --> 让路: 环境变化
    让路 --> 生效: 环境变回去
    让路 --> 删除: 上游合入，探测恒为假
    生效 --> 删除: 上游合入
    删除 --> [*]
```

「让路」是运行时行为，不是版本号判断——换到具备该能力的硬件上自动生效。

### 3.6 override：五步

```mermaid
flowchart TD
    A["输入：用户列出的一串目标<br/>每个是 模块.函数 ，可以带 kwargs"] --> B["① 把这些模块 import 进来"]
    B --> B2["import 的副作用是<br/>让模块里的 @override 完成注册"]
    B2 --> C["② 遍历配置树，算出每条 override<br/>究竟要改哪些节点"]
    C --> C2["产出一组 claim<br/>一个 claim = 这条 override + 这个节点 + 它的地址"]
    C2 --> D["③ 检查这组 claim 之间有没有冲突"]
    D --> E{"有冲突？"}
    E -->|有| X["报错并停下<br/>一个节点都不改"]
    E -->|没有| F["④ 逐个调用工厂函数<br/>旧 Config 进，新 Config 出"]
    F --> G["⑤ 把新 Config 装回树上原来的位置"]
    G --> H["输出：一行行替换日志<br/>谁把哪个地址换成了什么"]
```

| 步骤 | 在做什么 | 为什么是这个顺序 |
|---|---|---|
| ① 导入 | 把用户列出的模块 import 进来 | 注册是 `@override` 装饰器在 import 期干的事，所以「有哪些 override 可用」只有导入之后才知道 |
| ② 匹配 | 遍历配置树，找出每条 override 要改的节点 | 匹配规则两条：节点类型是不是这条 override 声明的 `target` 类（默认连子类一起算），以及节点地址是否命中它的 `fqns` 通配符 |
| ③ 查冲突 | 看这组 claim 之间有没有互相踩 | **必须在任何一次替换之前** |
| ④ 构造 | 把旧 Config 交给工厂函数，拿回新 Config | 工厂只负责「换成什么」，不负责「换哪儿」——换哪儿在 ② 就定好了 |
| ⑤ 写回 | `setattr` 回树上原来的位置 | 父容器与属性名在 ② 已经记在 claim 里，写回是纯机械动作 |

**claim 是这套流程里唯一需要新记住的概念**：它不是「一条 override」，而是「*这条 override*
要改 *这一个* 节点」。一条 override 用通配符命中 24 层，就产生 24 个 claim。
冲突检查检的是 claim，不是 override——所以两条 override 即使针对同一个 Config 类，
只要它们的 `fqns` 落在互不相交的层上，就不算冲突。这也是「同一个算子在一半层上 A/B 对比」
这种用法能成立的原因。

③ 为什么必须早于 ④：如果边遍历边替换，谁先谁后取决于用户写的目标顺序，冲突就变成一个
**时好时坏**的现象。先收集、再检查、最后统一改，冲突是一个确定性的错误，而且报错时
配置树还是完整的——一个节点都没动过。

### 3.7 两种冲突形态

区别不在于「谁和谁撞」，而在于**相等判定能不能发现它**。

```mermaid
graph TD
    subgraph A["形态一 · 两条 claim 落在同一个节点"]
        direction TB
        A1["override X<br/>claim layers.0.attention.inner_attention"] --> A3{"两个 FQN<br/>字符串相等"}
        A2["override Y<br/>claim layers.0.attention.inner_attention"] --> A3
        A3 --> A4["相等判定就能发现<br/>报错直观：narrow your fqns"]
    end
    subgraph B["形态二 · 一条落在另一条的祖先上"]
        direction TB
        B1["override X<br/>claim layers.0.attention"] --> B3{"两个 FQN<br/>字符串不相等"}
        B2["override Y<br/>claim layers.0.attention.inner_attention"] --> B3
        B3 --> B4["相等判定发现不了"]
        B4 --> B5["漏判的后果：父节点被整个换掉，<br/>子节点那条 claim 落在一个<br/>已经不在树上的对象上"]
        B4 --> B6["所以必须再判前缀包含：<br/>b.fqn 以 a.fqn + 点 开头"]
    end
```

> **冲突判定两条规则：相等，以及前缀包含。**
> 只判「两条 claim 的 FQN 是否相等」，会漏掉形态二——因为 `layers.0.attention` 和
> `layers.0.attention.inner_attention` 本来就是两个不同的字符串。判定必须补上第二条：
> **一方的 FQN 是不是另一方加上一个点之后的前缀**。两条都不成立，才算无冲突。

形态二在实际使用中很容易撞上，因为**上游自己也会提供 override**，而它可能认领一整块
（例如整个注意力块）。那一块同时是 inner attention 与 RoPE 两个节点的祖先，于是你针对
这两个节点的 override **都**要让位，不是只让一个。所以通用变换在施加自己的 override 之前，
得先看配置里已经存在哪些 override。

### 3.8 嵌套配置怎么换：derive

当一个节点需要连同它的子节点一起换，不能写两条 override（会触发形态二），
而是**一条 override 在父节点上，用 `derive` 就地构造子节点**。

```mermaid
graph LR
    subgraph BEFORE["替换前"]
        P1["InnerX.Config"] --> C1["XKernel.Config<br/>要 CUDA + 特定架构"]
    end
    subgraph AFTER["替换后 · 一条 override"]
        P2["替换后的 InnerX.Config"] --> C2["替换后的 Kernel.Config<br/>derive(cfg.kernel, ...)"]
    end
    BEFORE -->|"@override(target=InnerX.Config)"| AFTER
```

`derive` 的价值是**版本鲁棒**：目标 Config 以后长出新字段时，同名字段自动从 source 复制，
而不是静默退回默认值。工厂函数只写它真正要改的那几项。

---

## 4. 案例

### 4.1 注意力：一条链路，两个入口

```mermaid
flowchart LR
    subgraph UPSTREAM["上游只有两种 inner attention"]
        F["FlexAttention"]
        V["VarlenAttention"]
    end
    subgraph WHY["为什么都不能直接用"]
        F --> F1["编译版：编译器 lower 不了 document mask"]
        F --> F2["eager 版：实体化 O(T²) 分数矩阵 → OOM"]
        V --> V1["要 aten::_flash_attention_forward<br/>后端未提供该算子"]
    end
    subgraph OURS["两条 override，不同 target"]
        O1["从 flex 出发<br/>target=FlexAttention.Config"]
        O2["从 varlen 出发<br/>target=VarlenAttention.Config"]
    end
    F1 --> O1
    V1 --> O2
    O1 --> A["设备侧融合注意力<br/>packed 布局 + 逐文档因果掩码"]
    O2 --> A
```

两条 override 的 target 是不同的 Config 类，因此可以同时激活——上游不同 flavor 的默认节点不一样。

### 4.2 序列并行：为什么这条 override 必须*不*生效

```mermaid
flowchart TD
    A["序列并行开启"] --> B{"注意力节点是什么？"}
    B -->|"FlexAttention"| C["torch 有实现<br/>all-gather K/V"]
    B -->|"SDPA"| D["torch 有实现<br/>ring attention"]
    B -->|"VarlenAttention 或你的融合算子"| E["torch 没有钩子"]
    E --> F["上游抛 NotImplementedError"]
    C --> G["能跑 · eager flex，显存换正确性"]
    D --> G
    F --> H["硬失败"]
```

所以这条转换必须带一个前置条件：序列并行打开时直接返回，什么都不做。
**一个增量的适用条件，和这个增量本身一样，是设计的一部分。**

### 4.3 规则 R1：护栏不许 shim 掉

这是四个出口里唯一一条「看起来能做、但绝不能做」的，所以单独列成规则。

```mermaid
flowchart TD
    A["上游抛 NotImplementedError<br/>序列并行不支持这种注意力"] --> B{"这是什么？"}
    B --> C["护栏：它保护的机制在 torch 里，<br/>而 torch 只给 flex/SDPA 实现了它"]
    C --> D["shim 掉这个 raise 会怎样？"]
    D --> E["每张卡只在自己那段序列上算注意力"]
    E --> F["loss 静默算错，不报任何错"]
    F --> G["最坏的失败模式"]
    C --> H["正确做法：要么不触发它，<br/>要么实现它保护的机制"]
```

> **R1：一处 `raise` / `assert` 是护栏还是机制，决定它能不能被 shim。**
> 判断方法：**问这个 raise 保护的东西在哪里实现**。找不到实现，就是护栏。
> 护栏背后如果有未实现的机制，绕过它得到的不是「能跑了」，是「错得没有声音」。

---

## 5. 决策记录

### ADR-008（提案）门控探测根因，不探代理信号

- **Context**：每条增量都需要一个「什么时候不再需要我」的判据。最省事的写法是探一个容易拿到的
  相邻信号——某个包装没装、某个后端在不在。这类代理信号会在环境变化时**先于根因翻转**：
  装上一个包，探测变真，增量停止生效，而真正的限制还在，故障出现在离探测很远的地方。
- **Decision**：特性探测必须指向**根因本身**可观测的开关，哪怕那个开关更难拿到、是上游的私有属性。
- **Alternatives**：版本号比较——版本号不描述硬件能力，排除。试跑一次再回退——代价不可控，
  且会掩盖归因。
- **Consequences**：正面，环境或硬件变化时增量自动让路，不需要改代码，也不需要有人记得去改。
  负面，探测点常是上游私有属性，上游改名就会失效——因此所有探测集中在一个函数里，
  让它一处失效、一处修。

### ADR-009（提案）护栏不可 shim

- **Context**：上游为「这个组合我没实现」写的 `raise`，在使用者眼里和「这个组合被禁止了」
  长得一模一样。而 shim 恰好能一行删掉它，删完程序确实跑起来了。
- **Decision**：shim 不得移除 `raise` / `assert`，除非同时提供它挡住的那段实现。
- **Alternatives**：先删掉跑通再说——排除。护栏挡住的通常是「这条路上有一段计算没人做」，
  绕过去的结果不是报错，是一个安静的错误数值。
- **Consequences**：正面，杜绝一类无声的数值错误，这类错误在训练里可能几天才被发现。
  负面，某些能力要等你自己把机制实现出来，周期更长；这是应该付的代价。

---

## 6. 风险与对策

```mermaid
graph LR
    R1["shim 目标被上游挪走"] --> M1["apply_all 找不到目标就抛，<br/>错误信息里带 upstream 链接"]
    R2["shim 越积越多"] --> M2["数量是健康度指标，目标为零；<br/>每条挂 issue，上游合入即删"]
    R3["override 与上游 override 抢节点"] --> M3["collect-then-mutate + 祖先检查；<br/>通用变换先看已有 override 再施加"]
    R4["门控写错导致假绿假红"] --> M4["ADR-008 探根因；<br/>测量三态 + 归因规则做成数据表"]
    R5["静默降级污染性能数据"] --> M5["响亮降级 + provenance 表；<br/>无 provenance 的 benchmark 不收"]
    R6["结论基于单个模型的现象"] --> M6["stock 对照组<br/>__stock 模式，零 override"]
```

> **R6：没有对照组，就分不清「上游的问题」和「你的问题」。**
> 做法：让每个测量格子都能用一个后缀（例如 `__stock`）跑一遍**零 override** 的对照。
> 同一个用例，一次带你的增量、一次完全不带，两次都记录。没有这条，一个红格到底是
> 上游本来就不支持，还是你的替换弄坏的，只能靠猜——而猜错会把自己的问题写成硬件限制。
