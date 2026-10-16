# DGX Spark (GB) Multi-Node LLM Deployment & Tuning — Field Notes

English | [简体中文](README.zh-CN.md)

4× NVIDIA DGX Spark (GB10, aarch64) in a **switchless direct-attach fiber ring**, running DeepSeek-V4.1-Flash FP8 with 4-node TP4 inference. This document records **every measured number** — the working configs and the pitfalls alike.

> All figures are measured on real hardware, not extrapolated. Each tuning item comes as "before → after + acceptance anchor".

---

## Hardware & Environment

| Item | Value |
|---|---|
| Nodes | 4× DGX Spark GB10 (aarch64, 128GB unified memory each) |
| Interconnect | QSFP56 direct-attach fiber ring, **switchless**; MTU 9000 on every link |
| Control plane | Gigabit management LAN `192.168.88.0/24` |
| Data plane | RoCEv2 over fiber, `10.<a>.<b>.<d>`, d = node index |
| Stack | SGLang (not vLLM) |
| Model | DeepSeek-V4.1-Flash-UNCENSORED-FP8 (476GB, 48 shards flat on local disk) |

### Ring topology

```
dgx03 → dgx04 → dgx01 → dgx02 → dgx03
 rank0   rank1   rank2   rank3
```

Subnets are assigned as `10.<i>.<j>.<d>` so that every ring edge is a physically direct neighbor:

| Link | Subnets |
|---|---|
| dgx04 ↔ dgx01 | 10.0.2.x ↔ 10.0.3.x |
| dgx01 ↔ dgx02 | 10.0.3.x ↔ 10.0.0.x |
| dgx02 ↔ dgx03 | 10.0.0.x ↔ 10.0.1.x |
| dgx03 ↔ dgx04 | 10.0.1.x ↔ 10.0.2.x |

---

## 🔴 The most important finding: each GB10 QSFP port is internally **2 logical NICs**

The single highest-yield item in this whole tuning effort, and the easiest to miss.

```
$ ls /sys/class/infiniband/
rocep1s0f0  rocep1s0f1  roceP2p1s0f0  roceP2p1s0f1
                  ↑ second parallel link of the same physical port
```

**A single link only gives ~97–104 Gbps; aggregating both HCAs is required for the full ~200 Gbps.**

Measured `ib_write_bw`:

| Config | Bandwidth |
|---|---|
| Single link | 108.9 Gb/s |
| Dual-HCA aggregate | ~200 Gb/s |

### The mandatory four-piece env set (all or nothing)

```bash
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1     # list both HCAs
NCCL_CROSS_NIC=1                       # =0 and NCCL never spreads to the second link!
NCCL_SOCKET_IFNAME=<netdev_A>,<netdev_B>   # all three must cover both netdevs
TP_SOCKET_IFNAME=<netdev_A>,<netdev_B>
GLOO_SOCKET_IFNAME=<netdev_A>,<netdev_B>
NCCL_IB_ADDR_RANGE=10.0.0.0/22         # must cover both link subnets
```

**Key point**: env-only, effective immediately — **no NIC reconfiguration needed** (avoids the nmcli down/up incident path that kills NCCL).

### NCCL_MAX_NCHANNELS: 4 → 8 (the direct lever that doubled prefill)

```bash
NCCL_MAX_NCHANNELS=8
# drop NCCL_MIN_NCHANNELS (leave adaptation to NCCL)
```

SGLang prefill uses the Simple protocol and is bandwidth-bound; **pinning 4 channels halves the aggregated dual-HCA bandwidth**.

Measured steady-state prefill throughput:

| Config | tok/s |
|---|---|
| Baseline (4 channels) | 1070–1130 |
| After (8 channels) | **1565–1973** |

**≈1.5–1.8×**, root cause being the dual-HCA aggregate bandwidth.

### Unfixable pitfall: switchless rings require patched NCCL

Two NCCL copies coexist in the container: SGLang's pynccl gets the patched build via `LD_LIBRARY_PATH`, while torch still loads the stock one. **Stock has no switchless patch ⇒ no subnet-aware listener GID announcement at connection setup ⇒ wrong physical link picked ⇒ the whole communication layer collapses.**

Symptom (visible only with `NCCL_DEBUG=INFO`):

```
Call to ibv_modify_qp failed with 110 Connection timed out, dev rocep1s0f0:1,
local GID ::ffff:10.0.0.101, remote GID ::ffff:10.0.2.104
HINT: ... NICs are not cross-rail connected. ... set NCCL_CROSS_NIC=0
```

i.e. the wrong link is picked (dgx01's f0 connecting to dgx04's f0 — not directly wired; correct is f1↔f1).

Fix:

```bash
LD_LIBRARY_PATH=/nccl
LD_PRELOAD=/nccl/libnccl.so.2    # critical: patched build becomes the only runtime in the process
EP_SUPPRESS_NCCL_CHECK=1         # official switch for DeepEP's "dual NCCL runtime" assertion
NCCL_SKIP_TREE_CONNECT=1
NCCL_SWITCHLESS_RING_ONLY=1
```

`LD_PRELOAD` without `EP_SUPPRESS_NCCL_CHECK` hits
`AssertionError: Duplicate NCCL runtime found` (from `deep_ep/__init__.py: check_nccl_so()`).

**Success signature**: with `NCCL_DEBUG=INFO`, look for `ncclIbSenderQPsToRts: Set ECE`,
and `modify_qp` / `Duplicate NCCL` counts equal to **0**.

---

## The real semantics of memory / concurrency knobs

🔴 **Do not trust the documented "set a big pool" — in SGLang, `max_total_tokens` is only a cap.**

From `kv_cache_configurator.py:_apply_token_constraints`:

```python
user_limit = get_schedule().max_total_tokens
if user_limit is not None:
    if user_limit > token_capacity:
        logging.warning(f"max_total_tokens={user_limit} is larger than the profiled value "
                        f"{token_capacity}. Use the profiled value instead.")
    token_capacity = min(token_capacity, user_limit)   # ← cap only
```

The real lever is `MEM_FRACTION_STATIC` (static pool − weights − graphs = profiled capacity).
Pinning above profiled → warning + fall back; pinning below → pool = pin (smaller).

### Measured calibration (per rank)

| `MEM_FRACTION_STATIC` | `available_bytes` | profiled `full_token` | headroom outside pool |
|---|---|---|---|
| 0.80 | 9.10 GB | 5,450,xxx | 16.15 GB |
| 0.82 | — | ~6.2M | 15.08 GB |
| 0.86 | **15.34 GB** | 9,204,224 | 6.55 GB |

Each +0.06 fraction ≈ +6.24 GB budget ≈ +3.75M tokens. **Not linear** — graph capture / fixed
overheads eat part of it; **always trust the boot log**.

⚠️ At 0.86 only 6.55GB remains outside the pool; prefill transients (indexer + allocator
reallocating every step) take the slow path in shallow water. **0.80–0.82 is the sweet spot.**

---

## Operational notes

### Distributing images across nodes: never `docker save/load`

```bash
ctr -n moby images export - <ref> | ssh <target> "ctr -n moby images import -"
```

On this cluster `docker save` is broken (layers GC'd from the containerd content store) and
produces empty shells with "0 layers" (`docker images` shows 8kB); install looks fine and the
container only explodes at start. **After distributing, always verify with a real `docker create`**,
never just `docker inspect`.

### worker-first start / master-first stop

The wrong restart order deadlocks the master GPU.

### Pre-flight static checks (saves a 10-minute retry)

1. **Deployed files must be md5-identical on all 4 nodes** (scripts + `.env`)
2. **Dry-render** the docker run command (values with spaces must survive word-splitting)
3. **Env diff vs upstream `start.sh`** — never copy by hand

Measured cost of a missing env var: 3 missing vars → **Engram packing silently disabled**
(`packed=False`, two reads per miss).

### Startup latency

Container start → API ready ≈ **9–11 minutes** (parent loads 48 shards → engine subprocess → kernel autotune
→ KV pool → draft graph capture → smoke + warmup).

---

## Acceptance gate (all required)

1. `docker inspect` on all 4 nodes → `Running=true Restarts=0 Health=healthy`
2. `modify_qp` / `Duplicate NCCL` / `WorkerProc failed` counts = 0
3. `curl :8000/v1/models` → correct name, `max_model_len=1048576`
4. Real inference `19+23 → 42`, `finish_reason=stop`
5. Engram packing active: `packed=True` and `lookups == reads` (1:1)
6. **All 4 nodes' GPUs engaged**: under concurrent generation `nvidia-smi` shows 90–95% on all four

---

## Debugging discipline

- **Generic errors are second-order effects**: `remote process exited or there was a network error`
  tells you nothing; turn on `NCCL_DEBUG=INFO` and read the real cause (`ibv_modify_qp` family).
- **ctypes-based NCCL probes lie** (when passing structs without `argtypes`): plain
  `ncclCommInitRank` reports a 0.0.0.0 bootstrap failure while torch's PG in the same process succeeds.
- **Doc timestamps are UTC**: check timezones first (once read 8-hour-old logs as "just now").
- **A/B comparison beats probing**: diff the working stack (vLLM) against the crashing one (SGLang) on the same ring; it points straight at `LD_PRELOAD` and channel count.

---

## Related documents

- [SGLang hybrid SWA cache tuning](sglang-swa-hicache-tuning.md) — prefix-cache hit-ratio tuning (highlight)
- [HiCache 3-tier caching](hicache-l3-tuning.md) — L1/L2/L3 tiers and the all-or-nothing limitation
- [Troubleshooting index](troubleshooting.md) — symptom → root cause
