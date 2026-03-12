# CURRENT STATUS

Date: 2026-03-12
Repo: `/Users/wei/work/spark/leanstack`  
Remote deploy root: `/home/pto/lean`  
Active remote repo: `/home/pto/lean/repo`

## 1. 一句话总结

`leanstack` 已完成“固定模型 + 固定芯片 + 固定执行合同”的官方对比路径收敛：`Qwen/Qwen3-1.7B-Base + BF16 + GB10/sm_121 + decode_64_256 + packed appliance`，并将 compare/UI 与 remote benchmark 统一为 strict-contract 模式，用于支撑后续 DSA 上的 VIS 论证。

当前数据分层解释：
- **官方 fixed-contract 路径**：packed appliance 与 warmed vLLM 接近，尚未形成稳定的 `+30%` 决定性优势。
- **探索性路径**（例如 dual-model speculative）可以出现 `+30%`，但不作为 DSA VIS 主结论依据。

## 2. 当前最终目标

当前项目不再是“做一个比 vLLM 更小的 runtime”，而是：

- 选定固定模型：`Qwen/Qwen3-1.7B-Base`
- 选定固定硬件：`GB10 / sm_121`
- 选定固定精度：`BF16`
- 选定固定 bucket contract：例如 `decode_64_256`
- 选定固定服务形态：`leanpack + leanserve` appliance

在这个前提下，探索：

- 是否可以把兼容性税从 runtime 挪到离线 agent 生成和打包阶段
- 是否可以用更少的软件栈复杂度，换来更高的硬件效率
- 是否可以在这个固定合同下，稳定显著超过 warmed `vLLM`
- 是否可以把这套方法迁移到自研 DSA，并论证“VIS 是 agent 时代算力使能的核心边界”

## 3. 到目前为止的全部工作，按阶段整理

### Phase 0: 仓库、远端目录、基本工程骨架

完成内容：

- 新仓库命名为 `leanstack`
- 本地仓库位于 `/Users/wei/work/spark/leanstack`
- 已推送到 GitHub: `github.com/wshi/leanstack`
- 远端部署根目录固定为 `/home/pto/lean`
- 远端目录结构已建立：
  - `/home/pto/lean/repo`
  - `/home/pto/lean/models`
  - `/home/pto/lean/packed`
  - `/home/pto/lean/benchmarks`
  - `/home/pto/lean/artifacts`
  - `/home/pto/lean/logs`
  - `/home/pto/lean/tmp`

设计产物：

- 项目 thesis、架构、阶段计划、reference、benchmark protocol 等文档已系统化写入 `docs/`
- 全英文 skill 已完成

### Phase 1: cuTile / TileIR / remote bring-up

完成内容：

- 远端 DGX Spark / GB10 机器接入开发循环
- `remote.sh` 链路打通
- `cuTile` smoke 跑通
- 远端生成过 `.cutile`、`tilebc`、`tileiras`、`cubin`、SASS 等 artifact
- BF16 / FP8 / FP4 precision gate 已建立并实测

当前结论：

- `BF16` 是当前 public `cuTile` 路径下在 `sm_121` 上可工作的主精度
- `FP8` 当前 public path 能到编译器，但 TileIR verification 仍失败
- `FP4` 当前 public `cuda.tile` frontend authoring surface 不完整，因此主线已放弃 FP4

相关文档：

- `docs/PRECISION_GATES.md`
- `docs/FP4_COMPILER_GATE.md`
- `docs/REMOTE_VALIDATION.md`

### Phase 2: 模型目标多轮收缩与重定向

模型路线经历了几次明显 pivot：

1. 最早设想是 GLM
2. 后来切到 `Qwen3-32B`
3. 然后为了吞吐和可比较性，尝试 `Qwen3-8B-FP4`
4. 因为 FP4 public cuTile path 被 gate 卡死，又切回 BF16
5. 最终收敛到当前活动目标：
   - `Qwen/Qwen3-1.7B-Base + BF16 + GB10`

每次 pivot 的原因：

- `Qwen3-32B` 在 GB10 上虽然能跑，但初期只有约 `2 tok/s`，几乎没有 benchmark 价值
- `Qwen3-8B-FP4` 理论上更有吞吐潜力，但 public cuTile FP4 path 当前不成立
- `Qwen3-1.7B-Base` 是当前最适合验证“固定模型 + 固定硬件 + agent 生成推理代码”这条 thesis 的起点

当前活动模型合同：

- Model: `Qwen/Qwen3-1.7B-Base`
- Precision: `BF16`
- Hardware: `GB10 / sm_121`
- Main profile: `decode_64_256`

### Phase 3: 从 HF 黑箱 runtime 转向 explicit runtime ownership

完成内容：

- 去掉了最早 `HF + device_map="auto"` 这类黑箱式 placement 依赖
- 建立了显式的：
  - 磁盘权重装载
  - CPU staging
  - GPU materialization
  - KV cache ownership
  - 单层 block forward / prefill / decode
  - 多层 stack probe
  - full runtime loop

代表性代码：

- `src/leanstack/runtime/qwen_explicit.py`
- `src/leanstack/runtime/kv_cache.py`
- `experiments/models/qwen_explicit_block_probe.py`
- `experiments/models/qwen_explicit_stack_probe.py`
- `experiments/models/qwen_explicit_runtime_loop.py`

阶段性结果：

- 先打通 layer/block probe
- 再打通 full semantic runtime loop
- 明确把语义控制权从 `transformers` 拿回到 `leanstack`

这一阶段的真实收益不是吞吐，而是：

- 拿回 execution ownership
- 拿回 KV ownership
- 拿回 layer semantics ownership

### Phase 4: benchmark-first，建立与 vLLM 的同机对比

完成内容：

- 建立 benchmark profile system
- 建立统一 JSON 输出格式
- 建立 OpenAI-compatible benchmark client
- 建立远端 benchmark 脚本
- 建立 compare UI

关键脚本：

- `scripts/remote_openai_backend_benchmark.sh`
- `scripts/remote_leanstack_benchmark.sh`
- `scripts/remote_vllm_serve.sh`
- `scripts/serve_compare_ui.py`
- `src/leanstack/compare_runner.py`

阶段性结论：

- 最早的 semantic runtime 明显慢于 warmed `vLLM`
- 仅仅“更简单”并不会自动更快
- 必须切到 benchmark-driven work，而不是继续做抽象层面的整理

### Phase 5: 热路径优化，semantic runtime 从 29.95 提升到 44+ tok/s

这一步是最重要的一轮性能逼近：

早期数值：

- semantic full loop 主 profile 约 `29.95 tok/s`

之后做过的有效优化：

- KV get 不再 `torch.cat`
- decode attention 改到更紧的 `SDPA/GQA` 快路径
- resident runner / multi-request reuse
- RoPE cache
- RMSNorm fast path
- QKV / gate-up / decode helper 收紧
- token_count=1 append 快路径
- fixed-length request fast path

吞吐演进大致为：

- `29.95 tok/s`
- `36.49 tok/s`
- `40.81 tok/s`
- `44.55 tok/s`
- `44.61 tok/s`

这一步的结论是：

- runtime glue 优化并不是没用
- 但它只能把语义路径推到逼近 `vLLM`
- 它无法给出你要的 `+30%`

### Phase 6: 引入 appliance 思路，做 `leanpack + leanserve`

这是项目路线最关键的一次重构。

重构前：

- runtime 还是 checkpoint-driven
- 权重布局仍然更接近通用框架世界

重构后：

- `leanpack`: 离线 serving artifact
- `leanserve`: resident appliance runtime

对应代码：

- `src/leanstack/pack.py`
- `src/leanstack/leanserve.py`
- `scripts/remote_leanpack_build.sh`
- `scripts/remote_leanserve_layout.sh`

当前 packed artifact 事实：

- remote packed artifact:
  - `/home/pto/lean/packed/Qwen__Qwen3-1.7B-Base`
- resident layout:
  - `resident_weights_bytes ≈ 4.06 GB`
  - `max_kv_cache_bytes ≈ 124.8 MB`
  - `resident_bytes ≈ 4.19 GB`

性能意义：

- packed appliance path 把主 profile 推到约 `46.25 tok/s`
- 当前 warmed `vLLM` 约 `46.06 tok/s`

这说明：

- appliance 路径是当前唯一已经证明可竞争的正式路径
- 但它只是“略赢”，不是“显著赢”

### Phase 7: 为了 30% 目标，做 exact speculative decode 原型

用户明确把目标提高到：

- 相对 warmed `vLLM` 至少 `+30%`

因此项目进入 speculative 路线：

- exact speculative decode prototype
- draft/verifier split
- packed artifact 上的 draft metadata
- acceptance / committed tokens per cycle / throughput 统计

对应代码：

- `experiments/models/qwen_explicit_runtime_loop.py`
- `src/leanstack/draft_head.py`
- `src/leanstack/runtime/qwen_explicit.py`
- `src/leanstack/runtime/kv_cache.py`
- `scripts/remote_fit_draft_head.sh`
- `scripts/remote_qwen_runtime_loop.sh`

这一步又分成三轮：

1. naive shared-head self-spec
2. auxiliary draft head
3. block verifier + decode-calibrated draft head

最终结论：

- same-model prefix/suffix self-spec 这条线已经被充分验证
- 即使 acceptance 很高，甚至 `100%`
- 吞吐也仍然不能把 packed non-spec appliance 明显甩开

最新的代表性结果：

- packed non-spec appliance:
  - `46.25 tok/s`
- same-model speculative, `draft=24`, `k=2`, auxiliary head:
  - `40.8 tok/s`
  - acceptance `≈ 90.7%`
- same-model speculative, `draft=24`, `k=4`, auxiliary head:
  - `43.9 tok/s`
  - acceptance `= 100%`

因此到 2026-03-10 的关键结论是：

- prefix/suffix same-model split 不足以达到 `+30%`
- 真正下一步应该是：
  - 引入一个 genuinely smaller draft artifact
  - 而不是继续微调 same-model split

## 4. 当前成立的核心技术结论

### 已经成立的事

- `leanstack` 已经不是纸面规划，而是可运行系统
- remote GB10 上的 BF16 cuTile bring-up 成立
- `Qwen/Qwen3-1.7B-Base` full runtime loop 成立
- `leanpack + leanserve` appliance 成立
- packed appliance 在主 profile 上已经可以和 warmed `vLLM` 正面比较
- compare UI 可运行

### 还没有成立的事

- 相对 warmed `vLLM` `+30%`
- fully cuTile-native end-to-end decisive kernel path
- 一个可以显著放大吞吐优势的 speculative / draft 方案
- FP4 public cuTile path

### 当前最重要的工程判断

如果继续追求 `+30%`，下一步不应该是：

- 再继续抠 semantic runtime glue
- 再继续抠 same-model prefix/suffix self-spec

下一步应该是：

- second packed draft artifact
- 或者 another genuinely smaller exact draft appliance

## 5. 当前最重要的性能数据

### 主 profile

Main comparison profile:

- `decode_64_256`
- exact prompt bucket = `64`
- decode length = `256`

### 当前对比基线

- warmed `vLLM`: `≈ 46.06 tok/s`
- packed `leanstack` appliance: `≈ 46.25 tok/s`

### 历史演进

- early semantic runtime: `≈ 29.95 tok/s`
- optimized semantic runtime: `≈ 44.61 tok/s`
- packed appliance: `≈ 46.25 tok/s`

### speculative 结果

- `draft=24`, `k=2`, auxiliary head:
  - `≈ 40.8 tok/s`
  - acceptance `≈ 0.907`
- `draft=24`, `k=4`, auxiliary head:
  - `≈ 43.9 tok/s`
  - acceptance `= 1.0`
- decode-calibrated shallow heads:
  - `draft=8`, `k=2`: `≈ 25.7 tok/s`
  - `draft=12`, `k=2`: `≈ 16.8 tok/s`
  - `draft=16`, `k=2`: `≈ 16.1 tok/s`

解释：

- speculative 现在已经不是“没跑通”
- 而是“跑通了，但算力账不对”

## 6. 当前关键文件与用途

### 核心 runtime

- `src/leanstack/runtime/qwen_explicit.py`
  - Qwen semantic runtime / decode / verify / logits / speculative helpers
- `src/leanstack/runtime/kv_cache.py`
  - static KV cache / paged KV cache / speculative cursor rollback

### packed appliance

- `src/leanstack/pack.py`
  - `leanpack` artifact manifest 与 packed tensor 元数据
- `src/leanstack/leanserve.py`
  - `leanserve` artifact loader 与 resident buffer layout

### benchmark 与 compare

- `scripts/remote_vllm_serve.sh`
  - 在远端后台启动 `vLLM`
- `scripts/remote_openai_backend_benchmark.sh`
  - 跑远端 vLLM/OpenAI-compatible benchmark
- `scripts/remote_leanstack_benchmark.sh`
  - 跑远端 leanstack benchmark
- `scripts/serve_compare_ui.py`
  - 启动本地 compare UI
- `src/leanstack/compare_runner.py`
  - UI 背后的 orchestration 逻辑

### speculative

- `src/leanstack/draft_head.py`
  - auxiliary draft head fit
- `scripts/remote_fit_draft_head.sh`
  - 在远端拟合并写入 draft head
- `scripts/remote_qwen_runtime_loop.sh`
  - 在远端直接跑 runtime loop / speculative smoke

## 7. 如何启动本地 compare UI

### 前台启动

```bash
cd /Users/wei/work/spark/leanstack
python3 scripts/serve_compare_ui.py --host 127.0.0.1 --port 8787
```

访问：

- `http://127.0.0.1:8787`

### 后台启动

```bash
cd /Users/wei/work/spark/leanstack
nohup python3 scripts/serve_compare_ui.py --host 127.0.0.1 --port 8787 >/tmp/leanstack-compare-ui.log 2>&1 & echo $!
```

验证：

```bash
curl -s http://127.0.0.1:8787/api/status
```

说明：

- UI 只是本地薄前端
- 真正的 benchmark 和 vLLM 启动都在远端完成
- UI 内部有 operation lock，不允许并发点击多个长操作

## 8. compare UI 的工作流和背后脚本

### `/api/status`

对应逻辑：

- `scripts/serve_compare_ui.py`
- `src/leanstack/compare_runner.py::check_remote_status`

检查内容：

- 模型 path file 是否存在
- 远端模型目录是否完整
- `model.safetensors` 是否存在
- 远端是否已有 `vLLM` ready

### `Start vLLM`

UI 点击 `Start vLLM` 背后执行：

- `src/leanstack/compare_runner.py::ensure_vllm_ready`
- 实际脚本：`scripts/remote_vllm_serve.sh`

手工等价命令：

```bash
cd /Users/wei/work/spark/leanstack
VLLM_VENV=/home/pto/lean/venv-vllm-cu128 \
PYTHON_DEV_ROOT=/home/pto/lean/tmp/pydev_probe/extracted \
MODEL_ID=Qwen/Qwen3-1.7B-Base \
SERVED_MODEL_NAME=qwen3-1.7b-base \
./scripts/remote_vllm_serve.sh
```

远端日志和 pid：

- log: `/home/pto/lean/logs/vllm_8000.log`
- pid: `/home/pto/lean/logs/vllm_8000.pid`

验证：

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

### `Run Side-by-Side`

UI 点击 compare 背后的顺序是固定的：

1. 先检查远端状态
2. 确保 `vLLM` ready
3. 跑一轮 `vLLM` benchmark
4. 停掉 `vLLM`
5. 再跑一轮 `leanstack` benchmark

原因：

- GB10 显存有限
- `vLLM` 与 `leanstack` 不应该同时常驻占显存

背后逻辑：

- `src/leanstack/compare_runner.py::build_comparison_payload`

## 9. 手工跑 vLLM 对比

### 第一步：启动远端 vLLM

```bash
cd /Users/wei/work/spark/leanstack
VLLM_VENV=/home/pto/lean/venv-vllm-cu128 \
PYTHON_DEV_ROOT=/home/pto/lean/tmp/pydev_probe/extracted \
MODEL_ID=Qwen/Qwen3-1.7B-Base \
SERVED_MODEL_NAME=qwen3-1.7b-base \
./scripts/remote_vllm_serve.sh
```

### 第二步：跑 vLLM benchmark

```bash
cd /Users/wei/work/spark/leanstack
MODEL_ID=Qwen/Qwen3-1.7B-Base \
MODEL_NAME=qwen3-1.7b-base \
SYSTEM_LABEL=vllm \
VARIANT_LABEL=openai \
PROFILE=decode_64_256 \
SKIP_REMOTE_SYNC=1 \
./scripts/remote_openai_backend_benchmark.sh
```

如果要换 prompt：

```bash
cd /Users/wei/work/spark/leanstack
MODEL_ID=Qwen/Qwen3-1.7B-Base \
MODEL_NAME=qwen3-1.7b-base \
SYSTEM_LABEL=vllm \
VARIANT_LABEL=openai \
PROFILE=decode_64_256 \
PROMPT_OVERRIDE="Explain why a fixed model-chip contract can reduce inference overhead." \
SKIP_REMOTE_SYNC=1 \
./scripts/remote_openai_backend_benchmark.sh
```

输出 artifact 在远端：

- `/home/pto/lean/benchmarks/vllm_openai_decode_64_256_*.json`

## 10. 手工跑 leanstack benchmark

### 10.1 语义 runtime 路径

这条路径适合快速 smoke，不是当前最优结果路径。

```bash
cd /Users/wei/work/spark/leanstack
MODEL_ID=Qwen/Qwen3-1.7B-Base \
PROFILE=decode_64_256 \
RUNTIME_MODE=semantic \
NUM_LAYERS=0 \
SKIP_REMOTE_SYNC=1 \
./scripts/remote_leanstack_benchmark.sh
```

### 10.2 packed appliance 路径

这才是当前“官方最佳结果”路径。

```bash
cd /Users/wei/work/spark/leanstack
PACK_DIR=/home/pto/lean/packed/Qwen__Qwen3-1.7B-Base \
MODEL_ID=Qwen/Qwen3-1.7B-Base \
PROFILE=decode_64_256 \
RUNTIME_MODE=semantic \
NUM_LAYERS=0 \
SKIP_REMOTE_SYNC=1 \
./scripts/remote_leanstack_benchmark.sh
```

如果要指定 prompt：

```bash
cd /Users/wei/work/spark/leanstack
PACK_DIR=/home/pto/lean/packed/Qwen__Qwen3-1.7B-Base \
MODEL_ID=Qwen/Qwen3-1.7B-Base \
PROFILE=decode_64_256 \
RUNTIME_MODE=semantic \
NUM_LAYERS=0 \
PROMPT_OVERRIDE="Explain why a fixed model-chip contract can reduce inference overhead." \
SKIP_REMOTE_SYNC=1 \
./scripts/remote_leanstack_benchmark.sh
```

输出 artifact 在远端：

- `/home/pto/lean/benchmarks/leanstack_semantic_decode_64_256_*.json`

## 11. 手工跑 packed appliance 的 direct runtime loop

如果你不想走 benchmark harness，而是想直接看 runtime loop：

```bash
cd /Users/wei/work/spark/leanstack
MODEL_ID=Qwen/Qwen3-1.7B-Base \
PACK_DIR=/home/pto/lean/packed/Qwen__Qwen3-1.7B-Base \
RUNTIME_MODE=semantic \
MAX_PREFILL_TOKENS=64 \
MAX_NEW_TOKENS=256 \
EXACT_PREFILL_BUCKET=1 \
IGNORE_EOS=1 \
./scripts/remote_qwen_runtime_loop.sh
```

## 12. 手工拟合 draft head

### prefill calibration

```bash
cd /Users/wei/work/spark/leanstack
KEY=draft24_proj_v1 \
DRAFT_LAYER_COUNT=24 \
CHUNK_TOKENS=128 \
MAX_CHUNKS=32 \
CALIBRATION_MODE=prefill \
SKIP_REMOTE_SYNC=1 \
./scripts/remote_fit_draft_head.sh
```

### decode calibration

```bash
cd /Users/wei/work/spark/leanstack
KEY=draft8_decode_v1 \
DRAFT_LAYER_COUNT=8 \
CHUNK_TOKENS=64 \
MAX_CHUNKS=32 \
CALIBRATION_MODE=decode \
DECODE_STEPS=32 \
SKIP_REMOTE_SYNC=1 \
./scripts/remote_fit_draft_head.sh
```

这会把 draft head 写回：

- `/home/pto/lean/packed/Qwen__Qwen3-1.7B-Base/draft-heads.safetensors`
- 同时更新：
  - `/home/pto/lean/packed/Qwen__Qwen3-1.7B-Base/manifest.json`

## 13. 手工跑 speculative runtime

### 当前最强 same-model split

```bash
cd /Users/wei/work/spark/leanstack
MODEL_ID=Qwen/Qwen3-1.7B-Base \
PACK_DIR=/home/pto/lean/packed/Qwen__Qwen3-1.7B-Base \
RUNTIME_MODE=semantic \
SPECULATIVE=1 \
DRAFT_LAYER_COUNT=24 \
PROPOSAL_LEN=4 \
DRAFT_HEAD_KEY=draft24_proj_v1 \
MAX_PREFILL_TOKENS=64 \
MAX_NEW_TOKENS=256 \
EXACT_PREFILL_BUCKET=1 \
IGNORE_EOS=1 \
./scripts/remote_qwen_runtime_loop.sh
```

这条路径当前的代表结果：

- throughput `≈ 43.9 tok/s`
- acceptance `= 100%`

注意：

- 这仍然低于 packed non-spec appliance 的 `≈ 46.25 tok/s`

## 14. 当前 compare UI 的重要注意事项

### 1. UI 是 smoke tool，不是 authoritative packed-appliance harness

当前 `src/leanstack/compare_runner.py` 在跑 `leanstack` 时没有显式传 `PACK_DIR`。  
因此：

- UI 更适合做体验型 side-by-side smoke
- 当前“正式最好结果”仍应以手工 `PACK_DIR=... ./scripts/remote_leanstack_benchmark.sh` 为准

### 2. `Run Side-by-Side` 会自动停掉 vLLM

这是故意的，不是 bug。原因是避免远端 GPU 显存冲突。

### 3. 不要并发点按钮

UI 内部有 operation lock：

- `Start vLLM`
- `Run Side-by-Side`

不能并发运行。

### 4. 首次 `Start vLLM` 会比较慢

因为需要：

- 检查旧进程
- 启动远端服务
- 等待 `/v1/models` ready

## 15. 当前远端关键路径

### 模型

- path file:
  - `/home/pto/lean/models/Qwen__Qwen3-1.7B-Base.path`
- resolved model dir:
  - `/home/pto/lean/models/Qwen/Qwen3-1___7B-Base`

### packed artifact

- `/home/pto/lean/packed/Qwen__Qwen3-1.7B-Base`

### logs

- vLLM log:
  - `/home/pto/lean/logs/vllm_8000.log`
- vLLM pid:
  - `/home/pto/lean/logs/vllm_8000.pid`

### benchmark outputs

- `/home/pto/lean/benchmarks/leanstack_*.json`
- `/home/pto/lean/benchmarks/vllm_openai_*.json`

## 16. 当前已知问题 / operational notes

- local `compare UI` 当前不是 packed-appliance authoritative path
- same-model self-speculative decode 不足以支持 `+30%`
- `FP4` public cuTile path 仍 blocked
- `FP8` public path 仍未变成可用主线
- remote sync 时会出现大量 macOS xattr / `.DS_Store` / `LIBARCHIVE.xattr...` warning，噪音很大，但当前不影响正确性
- 仓库工作树当前只有 `.claude/` 未跟踪，没有别的未提交业务代码

## 17. 到当前为止，最重要的最终判断

### 已经证明的事

- 这条“固定模型 + 固定硬件 + agent 生成 appliance”路线不是空想
- `leanpack + leanserve` 的确比单纯 checkpoint-driven semantic runtime 强
- `leanstack` 已经有能力在主 profile 上逼平 warmed `vLLM`

### 还没有证明的事

- `leanstack` 可以显著超过 warmed `vLLM`

### 下一步最合理的方向

不是：

- 再继续抠 runtime glue
- 再继续抠 same-model prefix/suffix speculative

而是：

- second packed draft artifact
- genuinely smaller draft appliance
- 保持 verifier 为 full packed appliance
- 用真实 FLOP asymmetry 去换 `+30%`

## 18. Phase 8: 双模型 speculative decode（当前开发中）

### 第一性原理分析

decode 的根本瓶颈是 **显存带宽**：每生成一个 token 需要读取全部模型权重。

- 模型权重: ~3.4 GB (1.7B BF16)
- GB10 带宽: ~273 GB/s
- 理论上限: 273 / 3.4 ≈ 80 tok/s
- 当前实际: ~46 tok/s → 57.5% 带宽利用率
- +30% 目标: ~60 tok/s → 75% 带宽利用率

same-model speculative 失败的根本原因：draft 用了 24/28 层，成本是全模型的 86%，
没有真正的 FLOP asymmetry。即使 acceptance=100%，每个 cycle 的开销几乎等于标准 decode。

### 解决方案：Qwen3-0.6B 作为 external draft model

选择 `Qwen/Qwen3-0.6B-Base` 作为 draft model：

- ~1.2 GB 权重（全模型的 35%）→ ~7.6ms per draft token
- 与 1.7B 共享 vocabulary / tokenizer → token 直接兼容
- 不需要 projection head — 每个模型有自己完整的 LM head
- 不需要训练或微调

### 吞吐预测

验证步骤的关键洞察：verification 是 bandwidth-bound，k 个 token 的 verify 成本
约等于 1 次 decode（读取相同的权重，只是多了 k 倍算术，但算力远未饱和）。

```
k=5, acceptance=70%:
  Draft:  5 × 7.6ms = 38.0ms
  Verify: 1 × 21.7ms = 21.7ms
  Total:  59.7ms for 4.5 tokens (5×0.7 + 1 bonus)
  Throughput: 75.4 tok/s → +63% over vLLM

k=4, acceptance=50% (pessimistic):
  Draft:  4 × 7.6ms = 30.4ms
  Verify: 1 × 21.7ms = 21.7ms
  Total:  52.1ms for 3.0 tokens
  Throughput: 57.6 tok/s → +25%
```

### 实现方案

代码已实现但待远端验证：

1. **新增 `run_semantic_stack_verify_tokens`** (`src/leanstack/runtime/qwen_explicit.py`)
   - 接收 raw token IDs，用 verifier 的 embedding table 嵌入并运行全层验证

2. **新增 `run_dual_model_speculative_request`** (`experiments/models/qwen_explicit_runtime_loop.py`)
   - 双模型独立 KV cache
   - Draft phase: 0.6B 模型自回归生成 k 个 proposal tokens
   - Verify phase: 1.7B 模型单次 forward 验证全部 k tokens
   - Accept/reject + cache rollback

3. **新增 `qwen-draft` model spec** (`src/leanstack/model_registry.py`)
   - Qwen3-0.6B-Base 的 geometry 和 metadata

4. **新增脚本**
   - `scripts/remote_fetch_draft_model.sh` — 下载 draft model
   - `scripts/remote_leanpack_build_draft.sh` — 打包 draft artifact
   - `scripts/remote_dual_spec_benchmark.sh` — 运行双模型 speculative benchmark

### 运行步骤

#### 第一步：下载 draft model

```bash
./scripts/remote_fetch_draft_model.sh
```

#### 第二步：打包 draft artifact

```bash
./scripts/remote_leanpack_build_draft.sh
```

#### 第三步：运行双模型 speculative benchmark

```bash
# 默认 k=5
./scripts/remote_dual_spec_benchmark.sh

# 自定义 proposal length
PROPOSAL_LEN=8 ./scripts/remote_dual_spec_benchmark.sh
```

#### 也可以通过 runtime loop 直接跑

```bash
MODEL_ID=Qwen/Qwen3-1.7B-Base \
PACK_DIR=/home/pto/lean/packed/Qwen__Qwen3-1.7B-Base \
DRAFT_PACK_DIR=/home/pto/lean/packed/Qwen__Qwen3-0.6B-Base \
DUAL_MODEL_SPECULATIVE=1 \
PROPOSAL_LEN=5 \
RUNTIME_MODE=semantic \
MAX_PREFILL_TOKENS=64 \
MAX_NEW_TOKENS=256 \
EXACT_PREFILL_BUCKET=1 \
IGNORE_EOS=1 \
./scripts/remote_qwen_runtime_loop.sh
```

### 与 same-model speculative 的关键区别

| | same-model split | dual-model |
|---|---|---|
| Draft cost | 86% of verifier | 35% of verifier |
| Hidden state sharing | Draft hidden → verifier | None (token-level interface) |
| Projection head | Required (linear regression) | Not needed |
| Acceptance rate | High (same model) | Medium (different model) |
| Net throughput gain | ≤0% (proven insufficient) | Predicted +30-60% |

---

## Phase 9: Verify Batch Optimization & Remote Validation (2026-03-11, exploratory track)

### 关键发现：per-pass overhead 主导性能

通过远端 GB10 实测 profiling，发现了真正的性能瓶颈：

| | 带宽时间 | 实际时间 | Overhead |
|---|---|---|---|
| Verifier decode | 12.6ms | 21.9ms | 9.3ms (42%) |
| Draft decode | 4.4ms | 9.4ms | 5.0ms (54%) |

**Draft/Verifier 实际时间比为 42.9%（而非带宽理论的 34.6%）**。Overhead 来自 Python dispatch + CUDA kernel launch，28 层 × 10+ kernels/层 = 280+ kernel launches/pass。

### 关键优化：合并 verify batch

原始实现中，verify 阶段做了 **两次** verifier forward pass：
1. Verify batch: `[current_token, d₁, ..., d_{k-1}]` → k tokens
2. Bonus decode: `d_k` → 1 token → 获得 bonus prediction

**优化**：将 d_k 并入 verify batch → `[current_token, d₁, ..., d_k]` → k+1 tokens，一次拿到所有验证结果 + bonus。**省掉一整个 verifier decode pass (21.9ms/cycle)**。

### 实测结果

#### 自然文本（高接受率场景）

| k | Throughput | vs vLLM | Acceptance | Tokens/cycle | Cycles |
|---|-----------|---------|------------|-------------|--------|
| 3 | 62.2 tok/s | **+35.0%** | 87.7% | 3.61 | 71 |
| 5 | 61.8 tok/s | **+34.3%** | 82.8% | 5.12 | 50 |
| 7 | 69.3 tok/s | **+50.4%** | 92.5% | 7.31 | 35 |
| 10 | 71.1 tok/s | **+54.5%** | 93.2% | 10.24 | 25 |

**这组数据证明 dual-model speculative 作为探索方向可超过 +30%。**  
但它不属于当前 fixed-contract 官方结论口径（官方口径固定为单模型 packed appliance）。

#### 高多样性文本（低接受率场景）

| k | Throughput | vs vLLM | Acceptance |
|---|-----------|---------|------------|
| 5 | 47.2 tok/s | +2.6% | 58.1% |

接受率 < 60% 时几乎无增益。Speculative decode 的收益高度依赖 draft-verifier 的分布匹配度。

### 模型注册表修正

原始 `qwen-draft` ModelSpec 中 Qwen3-0.6B-Base 的几何参数来自错误来源，已修正为实际 config.json 值：
- hidden_size: 1024（非896）
- intermediate_size: 3072（非4864）
- num_attention_heads: 16（非14）
- num_key_value_heads: 8（非2）
- head_dim: 128（非64）

### 性能模型

```
cycle_time = k × t_draft + t_verify_batch(k+1)
throughput = tokens_per_cycle / cycle_time
tokens_per_cycle ≈ k × acceptance_rate + bonus_rate

其中:
  t_draft ≈ 9.4ms (0.6B single-token decode)
  t_verify_batch ≈ 21.9ms + Δ(k) (1.7B multi-token verify, Δ随k增长小)
```

### 结论（探索性）

**dual-model speculative 在自然文本场景中可实现 +30% 以上。** 关键技术栈：
1. 外部 draft 模型 (Qwen3-0.6B, 35% 权重) 提供真正的 FLOP 不对称
2. 合并 verify batch 消除冗余 forward pass
3. Qwen3 系列内 draft-verifier 匹配度高 (>80% 接受率)

**当前限制**：在高多样性/创意文本场景下，接受率降至 ~58%，增益趋近于零。后续优化方向包括 CUDA Graphs 降低 per-pass overhead、自适应 proposal length、或训练专门的 draft head。

**口径说明（2026-03-12 更新）**：以上结果属于 exploratory 路径，用于发现算法不对称来源；DSA VIS 主结论以 fixed-contract 官方路径为准。

这就是到 2026-03-12 为止，`leanstack` 的真实状态。
