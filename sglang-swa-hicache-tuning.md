# SGLang hybrid SWA 缓存调优：`/new` 后不秒出的根因与修复

> **一句话**：radix cache 一直是正常工作的精确 token 匹配；真凶是 hybrid SWA 模型的
> SWA 池**只有 40 条前缀的保热容量**，会话闲置约 1 小时后 tail 被 LRU 挤掉，整条命中作废。

本机实测：修复后**同样的 prompt 秒出**，523K 超长 ctx 也稳定命中。

---

## 1. 症状

- 新 session 第一条消息要 **12–14s** 才出 token
- SGLang 日志该批 `#cached-token: 0`（全冷重算 22K token）
- **同一分钟内别的会话却在命中 38 万 token**
- 用 vLLM 做后端时「偶尔能秒出」⇒ 容易误判成 SGLang 独有缺陷

---

## 2. 先排除误解：radix cache 就是精确匹配

社区说 SGLang radix cache 更优，**不是因为它模糊匹配**——是说它的树结构比 block-hash
粒度更细、能按 token 切分复用。匹配本身就是**逐 token 精确**的。

**决定性证据**：把 33,504 字符的 system prompt 只改一个日期（2 字符差异）：

```
改前 prompt 重放 : 14.06s   ← 冷
只改日期 2 字符  :  0.96s   ← 前 99.2% 全复用，只重算分歧点之后
同一 body 连发    :  0.41s
```

机制完全正常。所以问题在「树里那份前缀不在了」。

---

## 3. 根因：SWA 池 = 前缀缓存存活上限

### 源码注释原文（`pool_configurator.py::_resolve_swa_prefix_tails`）

> A radix-cached prefix is reusable only while its last sliding_window
> tokens still hold SWA slots, so **SWA capacity bounds how many cached
> prefixes stay hot**.

DeepSeek-V4 是 **hybrid SWA** 模型（启动日志 `Hybrid SWA model detected`），
缓存被拆成 full / SWA 两个组件，**SWA 侧是 hard cap**。

### 而且 `swa_prefix_tails` 默认是自动值

```python
def _resolve_swa_prefix_tails(self) -> int:
    prefix_tails = get_schedule().swa_prefix_tails
    if prefix_tails is not None:
        return prefix_tails
    if get_memory().disable_radix_cache:
        return 0
    max_running_requests = self.requested_max_running_requests_per_worker
    return 4 * max_running_requests if max_running_requests is not None else 0
```

`max_running_requests=10` ⇒ **全机只有 40 条前缀能保热**。多会话 + cron 一挤，
闲置 ~1h 的前缀 tail 必被 LRU 挤掉。

### 判据（近 6 小时 1,720 行 prefill 统计）

| 分类 | 行数 | `swa token usage` |
|---|---|---|
| `cached=0`（全冷） | 1336 | 均值 0.045，**最大 0.83** |
| `cached>0`（命中） | 384 | 均值 0.042，**最小 0.02，从不为 0** |

**`swa usage: 0.00` 的行 100% 是 `cached=0`。**

### 复现法（决定性实验）

同一份真实 prompt（33,504 字符 system + 25 个 tools ≈ 22,798 token）：

```
02:17:16  OLD-prev-session-prompt (闲置 1.6h):  wall=14.06s   ← 必冷
02:17:21  NEW-this-session-prompt:              wall=2.11s
02:17:25  OLD-again:                            wall=1.98s    ← 打一次就恢复
```

---

## 4. 修复

```bash
--swa-prefix-tails 160          # 40 → 160，hot 前缀 4 倍（治本）
--hicache-ratio 0.43            # host 池 7.64GB → 5.16GB，放出 2.5GB
MEM_FRACTION_STATIC 0.80 → 0.82 # 把释放的 RAM 吸进 device 预算
MAX_TOTAL_TOKENS 5400000 → 6200000
```

### 成本换算（改前必算）

每 SWA token ≈ **20,124 字节**（由 host 池分配日志反推：0.59GB / (92 页 × 43 层 ×
149,760 B × 320 tok/页)），device 同 layout 同价。

```
full : swa 字节比 ≈ 1,670.75 : 20,124 ≈ 1 : 12
```

**每条保热 tail 成本** = 384 token × 20,124 B ≈ **7.4 MB**。

### cap mode 公式

```python
# compute_swa_request_cap + _resolve_swa_cap_tokens
trailing  = window(128) + eviction_interval × draft_tokens(6) + page_size(256) = 384
swa_tokens = max_running_requests × (chunked_prefill_size + trailing)
           + prefix_tails × (window + page_size)
```

10 req / 1024 chunk / 40 tails → 14,080 + 15,360 = **29,440**（修复前）
10 req / 1024 chunk / 160 tails → 14,080 + 61,440 = **75,520**（修复后）

---

## 5. 实测结果

| 项 | 改前 | 改后 |
|---|---|---|
| `swa_tokens` | 29,440 | **75,520** ✅ |
| `prefix_tails` | 40 | **160** ✅ |
| full 池 | 5,399,808 | **6,199,808（+80 万）** |
| hicache host 池 | 7.64 GB | 5.16 GB |
| 池外余量 | 19.08 GB | 15.08 GB |

四台 `healthy` / `restarts=0` / NCCL 错误 `0` / smoke 42 通过。
**用户验证：「秒出了，有效」。**

验收锚：

```
DSV4 SWA sizing: mode=cap, swa_tokens=75520, request_cap+headroom=75520, prefix_tails=160
```

---

## 6. 关键认知：SWA 池大小与 ctx 长度**无关**

这是最容易搞错的一点。`compute_swa_request_cap` 算的是**每请求只留滑动窗口的尾部**：

一个 **523K** 的请求，SWA 只占它**末尾 384 token** —— 因为 SWA 层只看窗口内，
历史 token 的 SWA 状态无用。实测：

```
cached=523,264  ↗  swa token usage: 0.01   (= 755 token)
```

所以 `swa_prefix_tails` **限制的是「同时保热的前缀条数」，不是 ctx 长度**：

| 项 | 值 |
|---|---|
| 每请求 SWA 占用 | ~384 token（+ prefill chunk） |
| 每前缀可覆盖的 ctx 长度 | **无上限**（1K 或 1M 都一样） |
| 75,520 池子可保热条数 | **约 196 条**（扣并发槽后即 160） |

**长期依赖由 full 池的压缩 KV 承担**（本模型用压缩注意力 c4/c128），SWA 只管窗口内精算。

### 想再扩？

| `--swa-prefix-tails` | 保热条数 | 内存代价 |
|---|---|---|
| 160（当前） | 160 | — |
| 400 | 400 | +0.9 GB |
| 800 | 800 | +1.8 GB |

除非同时开几百个会话，**160 已足够**。

---

## 7. ⚠️ 两个陷阱

### 陷阱 1：`--swa-prefix-tails` 与 `--swa-full-tokens-ratio` **互斥**

显式传 ratio 会退回 ratio sizing 模式、**tails 失效**：

```python
def _resolve_swa_cap_tokens(self) -> Optional[int]:
    if self.operator_swa_ratio is not None:
        return None      # ← 退回 ratio 模式，cap/tails 全失效
```

只传 tails，别传 ratio。

### 陷阱 2：降 `hicache-ratio` **单独不会**让 device 池变大

HiCache host 池是在 **device 预算之外**从剩余 RAM 分配的
（boot 日志里 `Memory pool end` 早于 host 池分配）。

必须**同时抬 `MEM_FRACTION_STATIC`** 才能把释放的 RAM 吸进 device 池。

---

## 8. 一个容易被忽视的连带因素：tools 数组顺序

`tools` 排在 `messages` 之后，**数组顺序一变整个后半段前缀就断**：

```
tools 逆序（内容一字不改）: 8.70s
```

---

## 附：完整复现脚本

```python
import json, time, urllib.request

tools = json.load(open("req.json"))["tools"]      # 真实请求体里的 tools
sysp  = open("system_prompt.txt").read()          # 真实 system prompt

body = {"model": "deepseek-v4.1-flash-uncensored", "reasoning_effort": "low",
        "messages": [{"role": "system", "content": sysp},
                     {"role": "user", "content": "OK"}],
        "tools": tools, "max_tokens": 2, "temperature": 0.0}

def send(tag):
    req = urllib.request.Request("http://<head>:8000/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    print(f"{tag}: wall={time.time()-t:.2f}s pt={r['usage']['prompt_tokens']}")

send("cold")          # 期望 12-14s
send("immediate")     # 期望 <0.5s  ← 证明机制正常
# 隔 1h+ 再跑 → 若回到 12s 且日志 cached=0，即本问题
```

判定：对照 SGLang 日志的 `#cached-token` 与 `swa token usage`。
**`swa usage: 0.00` 且 `cached=0` ⇒ 命中本问题。**