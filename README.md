# 天权 TIANQUAN

**世界顶级分布式训练 × 高效推理服务统一内核** — 纯 NumPy、CPU 可跑、可复现、可验证

作者：**晨星 (Chenxing)** · License: Apache-2.0

---

## 一、为什么做这个方向（技术选型的论证）

在做方向决策前，我先做了一次**缺口审计**：作者名下已有 300 个仓库，覆盖
自动微分、Transformer、经典 ML 全谱系、量化（GPTQ/AWQ/RTN）、RAG、Agent、
高效注意力、强化学习、扩散模型、因果推断、图学习、TDA……**算法层已高度饱和**。

但逐条扫描后，有一整层**完全空白**：

| 能力 | 现有 300 仓 | 说明 |
|---|---|---|
| 集合通信（Ring AllReduce / Reduce-Scatter / All-to-All） | ❌ 零覆盖 | 所有并行策略的地基 |
| 数据并行 DDP（含梯度分桶） | ❌ | PyTorch DDP 的核心工程优化 |
| ZeRO-1/2/3 分片 | ❌ | 大模型能训起来的关键 |
| 张量并行（Megatron 切分） | ❌ | 单卡放不下时的唯一解 |
| 流水并行（GPipe / 1F1B） | ❌ | 跨节点扩展的关键 |
| 推理服务系统层（PagedAttention / 连续批处理 / 投机解码） | ⚠️ 零散 | 只在概念层提过，无可运行实现 |

**零重写原则的第一条不是"复用别人的代码"，而是"不重复自己"。** 第 301 个
算法仓库毫无价值；一个把**系统层**做成可运行、可断言、可复现的仓库才有。

第二个约束来自环境：本机 **AMD 16 核 / 16 GB / 无 GPU**。这恰恰是优势——
GPU 方案的性能数字**无法被第三方复现**（需要特定显卡、驱动、CUDA 版本），
而 CPU 上的并行加速比、通信量、内存占用**任何人都能跑出来核对**。
对"可验证"这个目标而言，CPU 不是妥协，是更硬的证据链。

因此方向定为：**把分布式训练与推理服务的系统层，在 CPU 上做成最小可信实现，
并把每一条教科书结论变成可断言的数字。**

---

## 二、架构

```
┌──────────────────────────────────────────────────────────────┐
│ L5 eval/      基准 harness（确定性 / 中位数 / 环境指纹 / JSON） │
├──────────────────────────────────────────────────────────────┤
│ L4 serve/     PagedAttention · 连续批处理 · Radix 前缀缓存      │
│               · 投机解码 · 增量推理引擎                        │
├──────────────────────────────────────────────────────────────┤
│ L3 train/     反向模式自动微分 · GPT(RMSNorm+RoPE+GQA+SwiGLU)   │
│               · AdamW / Muon · 训练循环                       │
├──────────────────────────────────────────────────────────────┤
│ L2 parallel/  DDP · ZeRO-1/2/3 · 张量并行 · 流水并行(1F1B)      │
├──────────────────────────────────────────────────────────────┤
│ L1 comm/      通信后端(线程/共享内存) + 带宽最优集合通信         │
├──────────────────────────────────────────────────────────────┤
│ L0 core/      数值稳定原语（logsumexp / RoPE / RMSNorm ...）    │
└──────────────────────────────────────────────────────────────┘
```

**设计信条**

1. **零重写**：每个模块都明确标注它复刻的是哪篇论文/哪个系统，不发明算法。
2. **可验证**：每条结论配一条"另一条路径"的硬不变量，而不是靠肉眼。
3. **可复现**：纯 NumPy + CPU，固定种子，无网络、无 GPU、无权重下载即可全绿。

**关键抽象：算法与传输介质解耦。**
所有集合通信算法只依赖一个原语 `exchange(send_buf, src)`（发布 → 屏障 → 订阅 → 屏障），
后端负责提供它（线程 / 共享内存）。因此同一份 ring 算法既能在单进程多线程里做功能验证，
也能在真实多进程里跑性能基准。

---

## 三、关键技术选型与依据

| 模块 | 采用方案 | 文献 / 系统依据 | 选型理由 |
|---|---|---|---|
| AllReduce | Ring（大消息）/ Tree（小消息） | Patarasuk & Yuan 2009；Horovod (Sergeev & Del Balso 2018) | Ring 达**带宽下界** 2(N-1)/N·S；Tree 延迟 O(log N)，小消息更优 |
| DDP | 梯度分桶 + 就绪触发 | PyTorch DDP (Li et al. 2020) | 小消息延迟主导，分桶把 N 次小通信压成 1 次大通信 |
| ZeRO | 优化器/梯度/参数三级分片 | Rajbhandari et al., SC 2020 | 单卡内存 16P → 4P/N，论文 Table 1 已成本仓库硬断言 |
| 张量并行 | Column/Row 切分 | Megatron-LM (Shoeybi et al. 2019) | 每层只需 **2 次 AllReduce**，通信量最低 |
| 流水并行 | 1F1B (PipeDream-Flush) | Narayanan et al., ICML 2021；GPipe (Huang et al. 2019) | 与 GPipe bubble 相同，但峰值激活 O(m)→O(p) |
| KV 缓存 | PagedAttention + 写时复制 | vLLM (Kwon et al., SOSP 2023) | 消灭预分配碎片；共享前缀时 beam search 内存降到 1/k |
| 批处理 | 迭代级连续批处理 | Orca (Yu et al., OSDI 2022) | 先完成的请求立即出队，槽位零空转 |
| 前缀复用 | Radix 树 + LRU | SGLang RadixAttention (Zheng et al. 2024) | 多轮对话/系统提示自动复用，命中率可量化 |
| 投机解码 | 接受-拒绝采样 | Leviathan et al., ICML 2023 | **定理保证**输出分布与目标模型完全一致，无损加速 |
| 优化器 | AdamW / Muon | Loshchilov & Hutter 2019；Jordan et al. 2024 | Muon 用 Newton-Schulz 正交化更新，Transformer 上常优于 AdamW |
| 模型 | Pre-RMSNorm + RoPE + GQA + SwiGLU | LLaMA/Qwen2 系列标配 | 当前解码器事实标准 |

---

## 四、相对现有方案的改进点（诚实版）

本项目**不声称**在绝对性能上超越 vLLM / DeepSpeed / Megatron —— 那是 C++/CUDA
工程体量的差距。它的增量价值在四处：

1. **把"实现了"升级为"证明了"。**
   多数实现只保证结果"看起来对"；这里 **Ring AllReduce 的每 rank 发送字节数被
   断言等于理论下界** `2(N-1)/N·S`（9 组配置全部精确相等），
   ZeRO 内存表逐格对齐论文 Table 1，1F1B 的 bubble 与理论值 `(p-1)/(m+p-1)`
   实测误差为 0，投机解码的输出分布过 TVD 统计检验。

2. **算法—后端解耦的单一原语。**
   所有集合通信只需 `exchange()` 一个同步原语，线程后端与共享内存后端共享
   同一份算法代码。因此功能验证（快）与性能基准（真实跨进程）跑的是同一套逻辑。

3. **无 GPU 也能复现的完整证据链。**
   vLLM/DeepSpeed 的数字依赖特定硬件；本仓库 `git clone && pip install -e .`
   即可在任何机器上跑出同一组结论，环境指纹随报告落盘。

4. **把教科书结论变成可运行断言。**
   "1F1B 省内存"、"投机解码无损"、"ZeRO-3 内存 1/N" 这些话在论文里是公式，
   在这里是 `pytest` 里的一条 assert。

---

## 五、实测性能数据

**环境**：Windows 11 · AMD 16 核 @ 3.80 GHz · 16.4 GB · Python 3.13.14 · NumPy 2.5.4
（全部数字由 `benchmarks/run_all.py` 实测落盘，见 `benchmarks/results/report.json`）

### 5.1 集合通信：Ring 通信量精确等于理论下界

| world_size | 消息 | Ring 实测通信量 | 理论下界 2(N-1)/N·S | 一致 |
|---|---|---|---|---|
| 2 | 8 KiB | 8,192 B | 8,192 B | ✅ |
| 2 | 8 MiB | 8,388,608 B | 8,388,608 B | ✅ |
| 4 | 512 KiB | 786,432 B | 786,432 B | ✅ |
| 4 | 8 MiB | 12,582,912 B | 12,582,912 B | ✅ |
| 8 | 8 MiB | 14,680,064 B | 14,680,064 B | ✅ |

（9 组配置全部精确相等，无一例外）

**Ring vs Tree 的取舍被实测验证**（这正是 `allreduce(algo="auto")` 的选路依据）：

| 场景 | Ring | Tree | 结论 |
|---|---|---|---|
| N=8，8 KiB（小消息） | 44.29 ms | **15.71 ms** | Tree 快 **2.8×**（延迟主导） |
| N=8，8 MiB（大消息） | **14.68 MB** 通信量 | 25.17 MB | Ring 省 **41.7%** 带宽 |

### 5.2 DDP 训练弱扩展（每 rank 4×16 token，三轮取中位数）

| world_size | 墙钟 | 聚合吞吐 | 扩展效率 | 总通信量 |
|---|---|---|---|---|
| 1 | 0.159 s | 3,229 tok/s | 100.0% | 0 KiB |
| 2 | 0.205 s | 4,984 tok/s | **77.2%** | 4,756 KiB |
| 4 | 0.258 s | 8,040 tok/s | **62.2%** | 7,134 KiB |
| 8 | 0.588 s | 6,694 tok/s | 25.9% | 8,323 KiB |

**分桶调优的实测收益**（N=8，同一模型 76K 参数）：

| bucket_bytes | AllReduce 次数/步 | 聚合吞吐 | 扩展效率 |
|---|---|---|---|
| 16 KiB | 20 | 1,716 tok/s | 3.5% |
| 64 KiB | 12 | 3,110 tok/s | 6.4% |
| **4 MiB** | **1** | **6,809 tok/s** | **14.1%** |

> 桶从 64 KiB 调到 4 MiB，N=8 吞吐提升 **119%**，通信量降 42%。
> 这验证了 PyTorch DDP 把 `bucket_cap_mb` 默认设为 25 MB 的工程直觉。
>
> **为什么不线性扩展（诚实说明）**：模型只有 76K 参数，单步计算约 7 ms，
> 而 8 进程下 ring allreduce 要走 2(8-1)=14 步同步，且 8 个进程各自拉起
> NumPy 多线程 BLAS 争抢 16 核。小模型上通信/同步占比天然很高——
> 这不是实现的失败，而是**并行不是免费的**这一事实的量化呈现。
> （实测过限制 `OMP_NUM_THREADS=1`：单进程吞吐从 6,574 掉到 3,548，
> 反而更差，说明在本负载下 BLAS 多线程的收益大于超订损失。）

### 5.3 ZeRO 内存账本（1M 参数，fp32，Adam，N=8）

| 策略 | 参数 | 梯度 | 优化器状态 | 单卡合计 | 相对 DDP |
|---|---|---|---|---|---|
| DDP | 4.0 MB | 4.0 MB | 8.0 MB | **16.0 MB** | 100.0% |
| ZeRO-1 | 4.0 MB | 4.0 MB | 1.0 MB | 9.0 MB | 56.2% |
| ZeRO-2 | 4.0 MB | 0.5 MB | 1.0 MB | 5.5 MB | 34.4% |
| ZeRO-3 | 0.5 MB | 0.5 MB | 1.0 MB | **2.0 MB** | **12.5%** |

与 Rajbhandari et al. 2020 Table 1 逐格一致，由 `tests/test_parallel.py` 断言。

实测梯度同步（N=4，76K 参数）：ZeRO-1 26.25 ms / 891.8 KiB；
ZeRO-2 16.02 ms / 445.9 KiB；ZeRO-3 内存账本降至 297 KiB。

### 5.4 张量并行

| TP 度数 | 单次前向+反向 | 通信量/次 | loss |
|---|---|---|---|
| 1 | 13.27 ms | 0 KiB | 3.43138 |
| 2 | 18.59 ms | 128 KiB | 3.43138 |
| 4 | 17.06 ms | 256 KiB | 3.43138 |

**关键正确性**：三种 TP 度数下 loss **完全相同**（3.43138），
且前向 logits 与未切分单进程模型逐位一致（误差 < 1e-10，见测试 I4）。

> **诚实结论**：小模型上 TP 不但没加速，反而慢（18.59 vs 13.27 ms）——
> 因为每层 2 次 AllReduce 的通信开销超过了切分带来的计算节省。
> **TP 的真实价值是内存**（每卡只需 1/N 权重），是"单卡放不下"时的解法，
> 而非加速手段。这一点在实践中经常被误传。

### 5.5 流水并行：1F1B 峰值激活与 bubble

| 配置 (p, m) | GPipe 峰值激活/stage | 1F1B 峰值激活/stage | 激活节省 | bubble 实测 | bubble 理论 (p-1)/(m+p-1) |
|---|---|---|---|---|---|
| (4, 8) | 8 | 4 | 50% | **27.3%** | 27.3% |
| (8, 32) | 32 | 8 | **75%** | **17.9%** | 17.9% |
| (4, 32) | 32 | 4 | **88%** | **8.6%** | 8.6% |

bubble 实测值与理论公式**完全吻合**（离散事件模拟得出）。
两种调度 bubble 相同，差别全在内存：1F1B 把峰值激活从 O(m) 压到 O(p)。

### 5.6 推理服务内核

| 机制 | 实测结果 |
|---|---|
| **PagedAttention** vs 按 max_len 预分配 | 利用率 **91.6%** vs 53.7%（**1.71×**，48 条长度 8~64 的请求） |
| **连续批处理** vs 静态批处理 | 吞吐 **5.877** vs 4.085 tok/step（**+43.9%**）；槽位利用率 83.9% vs 58.3%；平均延迟 **30.2** vs 43.7 step |
| **RadixAttention** 前缀复用 | 命中率 **97.5%**，省下 98.4% 的 prompt prefill token |
| **投机解码** γ=2 | 接受率 88.9%，target 前向减少 **2.56×** |
| **投机解码** γ=4 | 接受率 86.6%，target 前向减少 **3.72×** |
| **投机解码** γ=8 | 接受率 85.5%，target 前向减少 **5.09×** |

投机解码的输出分布经 4000 样本 TVD 检验（< 0.03），与 target 自回归分布统计一致
—— 这验证了 Leviathan et al. 的分布一致性定理，即"加速无损"。

---

## 六、复现步骤

```bash
git clone https://github.com/CJX0712/tianquan.git
cd tianquan

python -m venv .venv && .venv/Scripts/activate     # Windows
# source .venv/bin/activate                        # Linux / macOS

pip install -e ".[dev]"

# 1) 核心不变量自检（秒级，不需要 pytest）
tianquan selftest

# 2) 单元测试 + 不变量验证
python -m pytest -m "not slow" -q          # 91 项
python -m pytest -m "mp" -q                # 5 项真实多进程 (spawn)

# 3) 实测基准（约 75 秒）→ benchmarks/results/report.json
python benchmarks/run_all.py

# 4) 端到端示例
python examples/01_ddp_training.py         # 多进程 DDP 扩展率
python examples/02_zero_and_parallel.py    # ZeRO 内存 / TP 等价 / 流水调度
python examples/03_serving.py              # 服务内核全家桶

# 5) 训练与推理
tianquan train --steps 100
tianquan serve --n 16
```

**零外部依赖**：核心只依赖 NumPy。无网络请求、无模型权重下载、无 GPU 要求。

---

## 七、仓库结构

```
tianquan/
├── src/tianquan/
│   ├── core/numerics.py       L0 数值稳定原语（logsumexp/RoPE/RMSNorm...）
│   ├── comm/
│   │   ├── backend.py         通信后端：线程 / 共享内存（发布-订阅语义）
│   │   ├── collectives.py     Ring/Tree AllReduce · ReduceScatter · AllGather
│   │   │                      · All-to-All · Broadcast
│   │   └── launch.py          世界启动器（run_threads / spawn_world）
│   ├── parallel/
│   │   ├── ddp.py             DDP：梯度分桶 + 就绪触发 + 通信账本
│   │   ├── zero.py            ZeRO-1/2/3 + 论文内存表
│   │   ├── tensor.py          Megatron 张量并行（Column/Row 切分）
│   │   └── pipeline.py        GPipe / 1F1B + 离散事件模拟
│   ├── train/
│   │   ├── autograd.py        反向模式自动微分（可梯度检验）
│   │   ├── model.py           小 GPT：RMSNorm+RoPE+GQA+SwiGLU
│   │   ├── optim.py           AdamW / Muon / SGD
│   │   └── trainer.py         训练循环与合成数据
│   ├── serve/
│   │   ├── kv_cache.py        PagedAttention + 写时复制 + 预分配基线
│   │   ├── engine.py          增量推理引擎（prefill/decode）+ 采样
│   │   ├── scheduler.py       连续批处理 vs 静态批处理
│   │   ├── radix.py           RadixAttention 前缀树
│   │   └── speculative.py     投机解码（接受-拒绝采样）
│   ├── eval/bench.py          确定性基准 harness
│   └── cli.py                 tianquan 命令行
├── tests/                     96 项测试（含 5 项真实多进程）
├── benchmarks/                基准脚本 + 实测 JSON
├── examples/                  3 个端到端示例
└── .github/workflows/ci.yml   3 OS × 3 Python 版本
```

---

## 八、验证清单（96 项全绿）

| 层 | 不变量 | 验证方式 |
|---|---|---|
| L0 | softmax 行和 = 1、RoPE 保范数、logsumexp 不溢出 | 解析解对照 |
| L1 | **Ring 通信量 = 2(N-1)/N·S 下界**（9 组精确相等） | 字节计数器断言 |
| L1 | ring ≡ tree ≡ NumPy 归约；ReduceScatter+AllGather ≡ AllReduce | 逐位对比 |
| L1 | All-to-All 路由 (i,j) 语义、Broadcast 根传播 | 语义断言 |
| L2 | **DDP 梯度 ≡ 单进程 global batch 梯度**（N=1/2/4） | 1e-10 对比 |
| L2 | ZeRO 内存表对齐论文 Table 1；ZeRO-2 分片 ≡ 全局均值对应片段 | 逐格断言 |
| L2 | **张量并行输出 ≡ 未切分模型**（N=1/2/4） | 1e-10 对比 |
| L2 | 1F1B 峰值激活 ≤ p、GPipe = m、bubble ≈ (p-1)/(m+p-1) | 模拟断言 |
| L3 | **13 个算子的解析梯度 vs 中心差分**（~1e-7） | 有限差分 |
| L3 | GPT 端到端梯度检验、因果性扰动证明、训练收敛 | 有限差分 / 收敛 |
| L4 | **增量 decode ≡ 整段重算**；推理引擎 ≡ 训练模型 | 1e-10 对比 |
| L4 | PagedAttention 优于预分配；写时复制隔离性 | 内存统计 / 污染测试 |
| L4 | 连续批处理吞吐 ≥ 静态批处理 | 调度对比 |
| L4 | **投机解码分布 ≡ target 自回归分布** | 4000 样本 TVD |

---

## 九、已知局限（诚实声明）

1. **无 GPU**：所有数字均为 CPU 实测，绝对吞吐量无法与 CUDA 方案对标。
   本仓库对标的是**算法正确性与系统机制的量化**，不是硬件峰值。
2. **Python 实现**：自动微分与推理引擎为纯 NumPy，绝对性能比编译型框架低 1~2 个数量级。
   换 C++/CUDA 内核不会改变任何一条算法结论，但会让常数变小。
3. **TP 在小模型上不加速**（实测 18.59 vs 13.27 ms）——已在 5.4 明确说明原因。
4. **DDP 扩展效率在 N=8 时仅 25.9%** ——小模型通信占比过高所致，非实现缺陷；
   已给出分桶调优将吞吐提升 119% 的实测证据。
5. **共享内存后端每 rank 预留 64 MiB 槽位**

---

## 十、作者

**晨星 (Chenxing)** — GitHub [@CJX0712](https://github.com/CJX0712)

> 零重写 · 集大成 · 可运行 · 可复现 · 可验证

每一条性能数字都能被 `python benchmarks/run_all.py` 重新跑出来；
每一条正确性结论都能被 `python -m pytest` 重新证明。
