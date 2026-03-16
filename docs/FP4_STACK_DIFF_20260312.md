# FP4 Stack Diff Report (sm_121)

Date: 2026-03-12  
Scope: remote GB10 (`sm_121`), environment `/home/pto/venv-cutile`  
Objective: upgrade `cuda-tile` from `1.1.0` to `1.2.0`, rerun FP4 gate, and measure deltas

## Commands executed

1. Baseline inventory:

```bash
/bin/zsh scripts/remote_fp4_inventory.sh
```

2. Baseline FP4 gate:

```bash
/bin/zsh scripts/remote_fp4_gate.sh
```

3. Upgrade:

```bash
ssh pto@kb119856792y.vicp.fun -p 33402 \
  "bash -lc 'source /home/pto/venv-cutile/bin/activate && python3 -m pip install --upgrade cuda-tile==1.2.0'"
```

4. Post-upgrade inventory:

```bash
/bin/zsh scripts/remote_fp4_inventory.sh
```

5. Post-upgrade FP4 gate:

```bash
/bin/zsh scripts/remote_fp4_gate.sh
```

## Before vs after

| Metric | Before (`1.1.0`) | After (`1.2.0`) | Delta |
|---|---:|---:|---|
| `cuda_tile_version` | `1.1.0` | `1.2.0` | upgraded |
| `public_has_fp4_symbol` | `false` | `false` | no change |
| `datatype_mentions_fp4` | `false` | `true` | improved |
| `bytecode_mentions_fp4` | `false` | `false` | no change |
| `backend_sm121_available` | `true` | `true` | no change |
| gate `status` | `blocked` | `blocked` | no change |
| gate `blocker` | public frontend incomplete | public frontend incomplete | no change |

## Artifacts

- Baseline gate output:
  - `/home/pto/lean/artifacts/fp4-gate/fp4_gate_20260312T014643Z.json`
- Post-upgrade gate output:
  - `/home/pto/lean/artifacts/fp4-gate/fp4_gate_20260312T014759Z.json`

## Interpretation

1. `cuda-tile` upgrade to `1.2.0` succeeded.
2. FP4-related text appears in `_datatype.py` after upgrade (`datatype_mentions_fp4=true`).
3. The public dtype registry still does not expose an FP4 symbol.
4. The bytecode type registry still does not expose FP4 (`bytecode_mentions_fp4=false`).
5. Therefore, on the public `cuTile` frontend path, FP4 remains non-authorable for our gate definition, and status stays `blocked`.

## Practical conclusion for leanstack

- Official path remains:
  - `Qwen/Qwen3-1.7B-Base + BF16 + GB10/sm_121`
  - `cuTile -> TileIR -> cubin`
- FP4 on `sm_121` can still be explored only as an explicit wedge path (PTX/CUTLASS/custom-kernel), and should be tagged as `exploratory` rather than official contract evidence.

