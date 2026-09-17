# HiCache 三级缓存：L3 被 SWA 掐死（all-or-nothing）

> **结论**：HiCache 本身**有效**（实测从 NVMe 捞回 518,912 token），但它被 hybrid SWA
> 的 all-or-nothing 校验卡成瘸腿——**必须先扩 SWA 池，L3 才可用**。

---

## 1. HiCache 是什么

SGLang 的三级 KV 缓存：

```
L1  device KV 池   （GPU 统一内存）
L2  host 池        （统一内存，`--hicache-ratio` 控制大小）
L3  NVMe 文件      （`--hicache-storage-backend file`）
```

### DSV4 上只有一种活法

连撞三次才定性（**每次白烧 10 分钟：这些是启动期 raise，不是运行期**）：

| 尝试 | 结果 |
|---|---|
| `--hicache-size` | ❌ 被显式拒绝：`hybrid_pool_assembler.py:_deepseek_v4_num_host_pages` raise `"does not support --hicache-size; use --hicache-ratio instead"` |
| `buffer_only` 模式 | ❌ 被拒：DSV4 的 sidecar pools（C4/C128 压缩 region）没有 per-pool staging 路径，`buffer_mode/pipeline.py:validate_buffer_only_stack` raise `"does not support sidecar storage pools"` |
| `cache` 模式 + `--hicache-ratio` | ✅ **唯一可行** |

⚠️ *「统一内存 → 只能用 buffer_only」这个直觉在 DSV4 上是错的。*

⚠️ **排查提示**：这些 raise 在 `Attached hybrid pool stack to UnifiedRadixCache`
**之后**才触发——别被 "Attached ... pools=KV + SWA + DeepSeekV4 sidecars" 骗过。

---

## 2. 🔴 all-or-nothing：L3 的死穴

`unified_radix_cache.py::_check_hybrid_prefetch_result` 源码注释原文：

```python
# Hybrid cache state is all-or-nothing: every extra pool (SWA / Mamba / ...)
# must cover the same fetched prefix. If any pool falls short the whole
# prefetch result is unusable, so discard it and release everything.
```

驱逐时打印：

```python
logger.warning(
    "HiCache hybrid prefetch discarded req=%s completed=%d requested=%d "
    "kv_beliefs_kept_pages=%d", ...)
```

### 为什么这杀死了 L3

L3 里 KV 主体 **35,612 组文件**，而 SWA 组件只有 **13,788 个**。
SWA 设备池修复前仅 29,440 token（115 页）。

⇒ 长前缀的 SWA 组件留不下 ⇒ prefetch 整条作废 ⇒ **KV 部分明明命中了也不给用**。

### 实测证据（修复前）

prefetch 全程**只触发 7 次**，且 `loaded` 恒 0：

```
07:59:27  HiCache prefetch success completed=1024     matched=0     loaded=1024
07:59:28  HiCache prefetch success completed=3072     matched=0     loaded=3072
07:59:54  HiCache prefetch success completed=251648   matched=0     loaded=251648
10:56:21  HiCache prefetch success completed=39168    matched=0     loaded=39168
15:15:53  HiCache prefetch success completed=290304   matched=290304  loaded=0
15:20:30  HiCache prefetch success completed=1024     matched=0     loaded=1024
16:45:50  HiCache prefetch success completed=377600   matched=377600  loaded=0
```

`loaded=0` 且 `matched>0` ⇒ 是 **L2 在扛，L3 基本闲置**。因为 all-or-nothing
要求 KV + SWA 同时命中，而 SWA 侧只有 115 页可存，天然卡死。

---

## 3. 修复后：L3 真正开始供数据

扩 `--swa-prefix-tails` 到 160（SWA 池 29,440 → 75,520）后：

```
19:33:04  HiCache prefetch success completed=3072     matched=0  loaded=3072
19:33:24  HiCache prefetch success completed=168448   matched=0  loaded=168448
19:34:31  HiCache prefetch success completed=518912   matched=0  loaded=518912  ← 从 NVMe 实读 519K
19:34:31  Prefill batch  cached=518912  pend=0                       ← 立刻命中
19:34:44  Prefill batch  cached=519424  new=512  pend=0              ← 只算增量 512
19:36:16  Prefill batch  cached=522496  new=512  pend=0              ← 稳定每轮 +512
```

**`loaded` 从恒 0 变成主要通路**，且 523K 超长 ctx 稳定增量续写（`pend=0`，无重算）。

---

## 4. HiCache 的其它事实

- `--hicache-ratio` 语义：host 池页数 = ratio × device 池页数
  （`_deepseek_v4_num_host_pages`：`full_host_pages = int(device_full_pages * ratio)`）
- ⚠️ **host 池不在 device 预算内**：boot 日志里 `Memory pool end` 早于 host 池分配。
  所以降 ratio **单独不会**让 device 池变大，必须同时抬 `MEM_FRACTION_STATIC`。
- **MLA 下只有 rank0 写盘是 by design**（官方文档原文）：

  > For MLA models, **all ranks hold the complete and identical KV data**.
  > HiCache includes a dedicated optimization for MLA: **only one rank initiates
  > the write-back operation**.

  代码对应 `backup_skip = is_mla_model and tp_rank != 0`。
  所以四台里只有 rank0 的盘会有 .bin，rank1-3 恒空——**不是 bug**。

- ⚠️ **多机部署必须让 L3 真是共享的**，否则整个持久化层白建：MLA 下只有 rank0 写盘，
  若四台各挂本地盘，rank1-3 永远 miss，而 prefetch 前缀长度走 `all_reduce(op=MIN)`
  → **直接归零**。
  做法：**让 rank0（唯一写入方）自己做 NFS 服务端**，其余三台挂到同名路径；且
  **必须重启栈**（docker bind mount 是 `rprivate`，容器建好后再挂宿主机，容器里看不到）。

- **没有任何 TTL**：淘汰只看容量不看时间（实测跨 7 小时 22 分空闲仍命中 230,912 token）。
- 顶层 cap 由 `SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE` 控（本文档实机设 800G），淘汰走 LRU。
- **prefetch 超时公式**：`min(prefetch_timeout_max, base + per_ki_token × tokens/1024)`，
  默认 2s / 0.1s每1024token / 30s。

---

## 5. 🔴 `flush_cache` 是核选项

```bash
# ① 只拔 loop 的那些请求（首选，不动 radix tree）
curl -X POST http://127.0.0.1:8000/abort_request \
  -H 'Content-Type: application/json' -d '{"rid":"<具体 rid>"}'

# ② 全量 abort（仍不动 cache）
curl -X POST http://127.0.0.1:8000/abort_request -d '{"abort_all":true}'

# ③ flush_cache —— 最后手段
curl -X POST http://127.0.0.1:8000/flush_cache
```

实测事故：为清一段 loop 污染用 `flush_cache` → 全量清空 radix tree，
**把另一个客户端累积的 416,000 token 前缀缓存一起清了**（下一轮 `cached-token: 0`）。

- `abort_request` 的 schema（`AbortReq`）：`{rid, abort_all, finished_reason, abort_message, http_worker_ipc}`
- ⚠️ **空 rid + `abort_all=false` 会被静默忽略但返回 200**
  （日志 `Ignore abort_request with empty rid and abort_all=False`）
- `flush_cache` 在**有 running/waiting 请求时不执行**
- flush 后 radix tree 从零重建，期间所有客户端 `cached-token: 0`、TTFT 拉长——
  **体感像服务坏了，但是预期副作用**

---

## 6. 排查方法论（优先级高于技术细节）

**第一优先级 = 查官方文档**：

1. 官方文档（`docs.sglang.ai`、GitHub 官方 README、发布说明）
2. GitHub issues / 官方 blog
3. 社区文章
4. 都不行才进容器读源码

**为什么**：官方文档答「为什么这么设计、这是不是预期行为」；源码只答「代码怎么跑」。
实测教训：判断「HiCache 只有 rank0 写盘是不是 bug」翻源码花了 10+ 次 grep 仍只是推测，
**一次站内搜索就拿到官方原文一句话定性**。搜到官方原文即定性，不用再拿源码二次论证。