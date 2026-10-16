# DGX Spark (GB10) 多机 LLM 部署与调优实战

4× NVIDIA DGX Spark (GB10, aarch64) 组成 **switchless 光纤直连环**，跑 DeepSeek-V4.1-Flash
FP8 四机 TP4 推理。本文档记录**全部实测数据**——包括成功的配置和踩过的坑。

> 所有数字均为实机测量，非推算。每条调优给出「改前 → 改后 + 验收锚」。

---

## 硬件与环境

| 项 | 值 |
|---|---|
| 节点 | 4× DGX Spark GB10（aarch64, 统一内存 128GB/台） |
| 互联 | QSFP56 直连光纤环，**无交换机**；每条边 MTU 9000 |
| 控制面 | 千兆管理网 `192.168.88.0/24` |
| 数据面 | RoCEv2 over 光纤，`10.<a>.<b>.<d>`，d = 机器号 |
| 栈 | SGLang（非 vLLM） |
| 模型 | DeepSeek-V4.1-Flash-UNCENSORED-FP8（476GB，48 分片平铺本地） |

### 环拓扑

```
dgx03 → dgx04 → dgx01 → dgx02 → dgx03
 rank0   rank1   rank2   rank3
```

段号按 `10.<i>.<j>.<d>` 分配，保证每条 Ring 边都是物理直连邻居：

| 链路 | 段 |
|---|---|
| dgx04 ↔ dgx01 | 10.0.2.x ↔ 10.0.3.x |
| dgx01 ↔ dgx02 | 10.0.3.x ↔ 10.0.0.x |
| dgx02 ↔ dgx03 | 10.0.0.x ↔ 10.0.1.x |
| dgx03 ↔ dgx04 | 10.0.1.x ↔ 10.0.2.x |

---

## 🔴 最重要的发现：GB10 每个 QSFP 口内部是 **2 个逻辑 NIC**

这是全套调优里收益最大、也最容易被漏掉的一条。

```
$ ls /sys/class/infiniband/
rocep1s0f0  rocep1s0f1  roceP2p1s0f0  roceP2p1s0f1
                  ↑ 同一物理口的第二条平行链路
```

**只做单链只有 ~97–104 Gbps；双 HCA 聚合才跑满 ~200Gbps。**

实测 `ib_write_bw`：

| 配置 | 带宽 |
|---|---|
| 单链 | 108.9 Gb/s |
| 双 HCA 聚合 | ~200 Gb/s |

### 生效四件套（缺一不可）

```bash
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1     # 两个 HCA 都列
NCCL_CROSS_NIC=1                       # =0 时 NCCL 不会摊到第二链！
NCCL_SOCKET_IFNAME=<netdev_A>,<netdev_B>   # 三个都要覆盖两个 netdev
TP_SOCKET_IFNAME=<netdev_A>,<netdev_B>
GLOO_SOCKET_IFNAME=<netdev_A>,<netdev_B>
NCCL_IB_ADDR_RANGE=10.0.0.0/22         # 覆盖两链网段
```

**关键**：只改 env 即生效，**不必碰网卡**（避开 nmcli down/up 打挂 NCCL 的事故路径）。

### NCCL_MAX_NCHANNELS：4 → 8（prefill 翻倍的直接杠杆）

```bash
NCCL_MAX_NCHANNELS=8
# 删掉 NCCL_MIN_NCHANNELS（留给 NCCL 自适应）
```

SGLang 的 prefill 走 Simple 协议、带宽敏感，**固定 4 通道等于把双 HCA 的聚合带宽砍半**。

实测 prefill 稳态吞吐：

| 配置 | tok/s |
|---|---|
| 基线（4 通道） | 1070–1130 |
| 改后（8 通道） | **1565–1973** |

**≈1.5–1.8×**，根因就是双 HCA 聚合带宽。

### 修不了的坑：switchless 环必须打 patched NCCL

容器里同时存在两份 NCCL：SGLang 的 pynccl 经 `LD_LIBRARY_PATH` 拿到 patched 版，
而 torch 仍加载 stock。**stock 无 switchless 补丁 ⇒ 建连接时不做 subnet 感知的 listener
GID 宣告 ⇒ 选错物理线 ⇒ 通信层全崩**。

症状（只有 `NCCL_DEBUG=INFO` 才可见）：

```
Call to ibv_modify_qp failed with 110 Connection timed out, dev rocep1s0f0:1,
local GID ::ffff:10.0.0.101, remote GID ::ffff:10.0.2.104
HINT: ... NICs are not cross-rail connected. ... set NCCL_CROSS_NIC=0
```

即选错链路（dgx01 的 f0 去连 dgx04 的 f0——两者不直连；正确是 f1↔f1）。

修复：

```bash
LD_LIBRARY_PATH=/nccl
LD_PRELOAD=/nccl/libnccl.so.2    # 关键：patched 成为进程内唯一权威 runtime
EP_SUPPRESS_NCCL_CHECK=1         # DeepEP「双 NCCL runtime」断言的官方开关
NCCL_SKIP_TREE_CONNECT=1
NCCL_SWITCHLESS_RING_ONLY=1
```

只加 `LD_PRELOAD` 不加 `EP_SUPPRESS_NCCL_CHECK` 会撞
`AssertionError: Duplicate NCCL runtime found`（源自 `deep_ep/__init__.py: check_nccl_so()`）。

**生效标志**：`NCCL_DEBUG=INFO` 下出现 `ncclIbSenderQPsToRts: Set ECE`，
且 `modify_qp` / `Duplicate NCCL` 计数为 **0**。

---

## 内存 / 并发参数的真实语义

🔴 **不要相信文档里的「设大池子」——SGLang 里 `max_total_tokens` 只是上限。**

`kv_cache_configurator.py:_apply_token_constraints` 源码：

```python
user_limit = get_schedule().max_total_tokens
if user_limit is not None:
    if user_limit > token_capacity:
        logging.warning(f"max_total_tokens={user_limit} is larger than the profiled value "
                        f"{token_capacity}. Use the profiled value instead.")
    token_capacity = min(token_capacity, user_limit)   # ← 只是上限
```

真正的杠杆是 `MEM_FRACTION_STATIC`（静态池 − 权重 − graph = profiled 容量）。
pin 高于 profiled → 告警 + 回退；pin 低于 → 池 = pin（更小）。

### 实测标定（per rank）

| `MEM_FRACTION_STATIC` | `available_bytes` | profiled `full_token` | 池外余量 |
|---|---|---|---|
| 0.80 | 9.10 GB | 5,450,xxx | 16.15 GB |
| 0.82 | — | ~6.2M | 15.08 GB |
| 0.86 | **15.34 GB** | 9,204,224 | 6.55 GB |

每 +0.06 fraction ≈ +6.24 GB 预算 ≈ +3.75M token。**不是线性**——graph 捕获/固定开销
吃掉一部分，**以 boot 日志为准**。

⚠️ 0.86 时池外仅 6.55GB，prefill 瞬态（indexer + allocator 每步重分配）在浅水区走慢路径。
**0.80–0.82 是甜点区**。

---

## 启动编排要点

### 镜像跨机分发：绝不用 `docker save/load`

```bash
ctr -n moby images export - <ref> | ssh <目标> "ctr -n moby images import -"
```

本集群 `docker save` 已坏（containerd content store 里 layer 被 GC），产出「层数=0」的
空壳（`docker images` 显示 8kB），install 看着正常、起容器才炸。
**分发后必须 `docker create` 实测**，不能只看 `docker inspect`。

### worker-first 起栈 / master-first 停栈

重启顺序错会锁死 master GPU。

### 启动前静态核对（省一次 10 分钟）

1. **四台部署件 md5 必须一致**（脚本 + `.env`）
2. **干跑渲染** docker run 参数（含空格的值不能被 word-split 撕碎）
3. **env 差集核对**：与上游 `start.sh` 做差集，别手工抄

漏 env 的实测后果：漏 3 个变量 → **Engram 打包白打**（`packed=False`，每次 miss 两次读）。

### 启动耗时

容器起到 API 就绪 ≈ **9–11 分钟**（父进程 48 分片加载 → 引擎子进程 → kernel autotune
→ KV 池 → draft graph 捕获 → smoke + warmup）。

---

## 验收门（缺一不算通过）

1. 四台 `docker inspect` → `Running=true Restarts=0 Health=healthy`
2. `modify_qp` / `Duplicate NCCL` / `WorkerProc failed` 计数 = 0
3. `curl :8000/v1/models` → 名称正确、`max_model_len=1048576`
4. 真实推理 `19+23 → 42`，`finish_reason=stop`
5. Engram 打包生效：`packed=True` 且 `lookups == reads`（1:1）
6. **四机 GPU 都参与**：并发生成时 `nvidia-smi` 四台 90–95%

---

## 排障纪律

- **通配错误是二次效应**：`remote process exited or there was a network error` 什么都
  说明不了，必须开 `NCCL_DEBUG=INFO` 看 `ibv_modify_qp` 类真因。
- **`ctypes` 自建 NCCL 探针会骗人**（不带 `argtypes` 传 struct 时）：plain
  `ncclCommInitRank` 报 0.0.0.0 bootstrap 失败，而同进程内 torch 的 PG 成功。
- **doc 时间戳是 UTC**：排查先对时区（曾把 8 小时前的日志当「刚才」）。
- **对照组比探针有效**：同环上能跑的栈（vLLM）与崩的栈（SGLang）的差异点，直接指向
  `LD_PRELOAD` 与通道数。

---

## 相关文档

- [SGLang hybrid SWA 缓存调优](sglang-swa-hicache-tuning.md) — 前缀缓存存活率调优（重点）
- [HiCache 三级缓存](hicache-l3-tuning.md) — L1/L2/L3 缓存与 all-or-nothing 限制
- [排障手册](troubleshooting.md) — 症状 → 根因索引