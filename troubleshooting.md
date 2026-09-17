# 排障手册：症状 → 根因索引

按症状查表。每条给出**真因判据**和**怎么确认**，不靠猜。

---

## 一、推理性能类

### `/new` 后第一条消息要 12–14s 才出 token

| 项 | 内容 |
|---|---|
| **真因** | hybrid SWA 模型 SWA 池耗尽 → 缓存前缀 tail 被 LRU 挤掉 → 整条命中作废 |
| **判据** | SGLang 日志该批 `#cached-token: 0` **且** `swa token usage: 0.00` |
| **反例** | 若同时段别的会话在 `cached=` 大值命中 ⇒ 确认不是机制问题，是保热容量 |
| **修复** | `--swa-prefix-tails 160`（详见 [swa 调优](sglang-swa-hicache-tuning.md)） |

⚠️ 不要误判成「radix cache 不精确」。验证机制是否正常：**同一 body 连发第二次**，
若 <0.5s 则机制完全正常。

### prefill 吞吐只有上游的 1/3

| 项 | 内容 |
|---|---|
| **真因** | 单 HCA 链路（未做双 HCA 聚合）；或 `NCCL_MAX_NCHANNELS` 被固定为 4 |
| **判据** | `ib_write_bw` 单链 108.9 Gb/s vs 聚合 ~200 Gb/s |
| **修复** | 见 README「双 HCA 聚合四件套」+ `NCCL_MAX_NCHANNELS=8` |

⚠️ 另一个可能：`MEM_FRACTION_STATIC` 过高使池外余量过小，prefill 瞬态走 allocator 慢路径。

### GPU 98–100% 但客户端没发新请求（loop 症状）

| 项 | 内容 |
|---|---|
| **真因** | radix tree **放大**客户端 ctx 里的重复语义词 → 污染段成热点 → 沿同一路径复读 |
| **判据** | 单请求 `bytes>1MB / chunks>4000`（正常 ~20KB / ~70 chunks，差 20 倍） |
| **处置** | `abort_request` 先止血；**治本要清掉发起方的上下文** |

**最大的 loop 源往往是 agent 自身的并发子代理**：子代理 ctx 里全是
「skill/合并/去重/SKILL.md/重复」这类高度重复词面。规则：
批量重写重复词面的活**不派子代理、不过推理后端**，用本地脚本做。

---

## 二、启动 / 通信类

### worker 崩 `NCCL error: remote process exited or there was a network error`

| 项 | 内容 |
|---|---|
| **说明** | **这是二次效应，什么都说明不了** |
| **必做** | 开 `NCCL_DEBUG=INFO` 看真因 |
| **常见真因** | 选错物理链路：`ibv_modify_qp failed with 110`，local/remote GID 不构成直连邻居 |
| **修复** | patched NCCL（`LD_PRELOAD`）+ `NCCL_SKIP_TREE_CONNECT=1` |

### `AssertionError: Duplicate NCCL runtime found`

| 项 | 内容 |
|---|---|
| **真因** | 只加了 `LD_PRELOAD` 但没加 `EP_SUPPRESS_NCCL_CHECK=1` |
| **来源** | `deep_ep/__init__.py: check_nccl_so()` |
| **修复** | 加 `EP_SUPPRESS_NCCL_CHECK=1` |

### `ModuleNotFoundError: No module named 'loop_abort'`（起栈即炸）

| 项 | 内容 |
|---|---|
| **真因** | 上游 repo 更新引入了新 adapter 模块，但镜像内是旧快照、override 白名单没跟上 |
| **修复** | override 名单必须含 sitecustomize 依赖的**全部**模块；上游更新后先比对「镜像内 adapter/*.py vs repo」 |
| **静态验证法** | 容器内 `PYTHONPATH=${override}:${adapter} python3 -c "import <各依赖模块>"` |

### 镜像分发后起容器炸（`docker images` 显示 8kB）

| 项 | 内容 |
|---|---|
| **真因** | 集群 `docker save` 坏了（containerd content store 里 layer 被 GC），产出「层数=0」空壳 |
| **修复** | 一律 `ctr -n moby images export - \| ssh <目标> "ctr -n moby images import -"` |
| **验证** | 分发后**必须 `docker create` 实测**，不能只看 `docker inspect` |

### 启动期 raise：「does not support XXX」

DSV4 HiCache 的三个启动期 raise（**每次白烧 10 分钟**）：

| 参数 | 报错 |
|---|---|
| `--hicache-size` | `does not support --hicache-size; use --hicache-ratio instead` |
| `buffer_only` | `does not support sidecar storage pools` |
| `NCCL_MIN_NCHANNELS=4` | 半带宽 |

⚠️ 这些 raise 发生在 `Attached hybrid pool stack to UnifiedRadixCache` **之后**——
别被「Attached」骗过以为已经成功。

---

## 三、内存 / 池类

### `max_total_tokens=N is larger than the profiled value M. Use the profiled value instead.`

| 项 | 内容 |
|---|---|
| **说明** | **正常**，不是错误。`max_total_tokens` 只是上限，真杠杆是 `MEM_FRACTION_STATIC` |
| **源码** | `kv_cache_configurator.py:_apply_token_constraints` → `token_capacity = min(token_capacity, user_limit)` |
| **修复** | pin 设成略低于预期 profiled |

### 池外余量过小导致 prefill 变慢

| 项 | 内容 |
|---|---|
| **判据** | boot 日志 `available_gpu_mem` 低于 ~10GB（0.86 时仅 6.55GB） |
| **修复** | `MEM_FRACTION_STATIC` 降回 0.80–0.82 |

### 降 `hicache-ratio` 后发现 device 池没变大

| 项 | 内容 |
|---|---|
| **真因** | host 池在 device 预算**之外**分配（boot `Memory pool end` 早于 host 池分配） |
| **修复** | 必须**同时抬 `MEM_FRACTION_STATIC`** |

---

## 四、缓存类

### HiCache 一直不落盘 / rank1-3 目录空

| 项 | 内容 |
|---|---|
| **说明** | MLA 下只有 rank0 写盘是 **by design** |
| **官方原文** | *"only one rank initiates the write-back operation"* |
| **代码** | `backup_skip = is_mla_model and tp_rank != 0` |

### prefetch 明明命中却被丢弃

| 项 | 内容 |
|---|---|
| **真因** | `_check_hybrid_prefetch_result` 的 **all-or-nothing**：SWA/Mamba 组件缺额 ⇒ 整条作废 |
| **日志** | `HiCache hybrid prefetch discarded req=... completed=... requested=...` |
| **修复** | 扩 SWA 池 |

### 清 loop 污染把别人的缓存也清了

| 项 | 内容 |
|---|---|
| **真因** | 用了 `flush_cache`（全量清空 radix tree） |
| **实测** | 清掉了另一客户端累积的 **416,000 token** 前缀缓存 |
| **修复** | 优先 `abort_request`（逐 rid 或 `abort_all`），`flush_cache` 是最后手段 |

⚠️ `abort_request` 空 rid + `abort_all=false` **被静默忽略但返回 200**。

---

## 五、诊断工具

### 正确的 NCCL 探针

✅ 用 torch PG 建 4 机 all_reduce，秒级验证 fabric 与 patched NCCL 是否自洽。

❌ **`ctypes` 自建探针会骗人**（不带 `argtypes` 传 struct 时）：plain
`ncclCommInitRank` / `ncclCommInitRankConfig` 都报 0.0.0.0 bootstrap 失败，
而同进程内 torch 的 PG 成功。别拿它的结论当定论。

### 时区

docker logs 时间戳是 **UTC**。排查先对时区（曾把 8 小时前的日志当「刚才」）。

### 进程名别误读

容器内进程名 `sglang::scheduler_TP<n>_EP<n>` 的数字是 **node-rank**，
不是 TP size——别误判成「残留的 TP3 栈」。

---

## 六、操作纪律

1. **重启 = 昂贵交易**（~10 分钟 + 权重重载 + 全机 GPU 占用），未经同意绝不重启。
   能只改配置的下次生效就只改配置。
2. **变更有可核对证据**（md5 一致、干跑渲染、boot 日志行、验收门）再宣布完成。
3. **改 RoCE 网卡前必确认该机无 LLM 容器在跑**——nmcli down/up 会打断 NCCL 链路。
4. **LLM 前后端重启顺序错会锁死 master GPU**：先停 master 再停 worker，起则 worker-first。
5. **env 必须与上游做差集核对**，不要手工抄（漏 3 个变量曾导致 Engram 打包白打）。