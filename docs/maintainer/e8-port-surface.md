# E8 4-bit KV port — conflict surface inventory

Baseline `pre-e8-baseline` @76e7f2e; lineage official → Don-Chad → UDP (E8 codec source) → sergiuszm
(sm_89 retune + E8) → xkeyc (host prefix cache + YaRN). Local v3 history was rewritten, so there is
**no common ancestor** — every classification is a tree diff. Refs: xkeyc `93419f4d`, sergiuszm `aeeba414`, FORK `981b685e`.

## ① Patch surface (line counts = `git diff --stat`)

| path (upstream) | local v3 target | lines | disposition |
|---|---|---|---|
| `src/ops/kernel/e8_lattice.cuh`, `e8_root_codec.cuh` (A) | `src/ops/kv_cache/e8/` | +184/+733 | drop-in (one include retarget each) |
| `tests/ops/test_e8_root_codec.cu` (A) | `tests/ops/` | +212 | drop-in + ctest; own `main()`, links `ninfer_ops` only (upstream never registered ctest) |
| `tools/test_kv/{kv_oracle.py,test_e8_codec.{cuh,cu},verify_1m_retrieval.cu}` (A) | same | +1118 | drop-in; `kv_oracle.py` = host numpy cosine oracle (zero-GPU fidelity) |
| `softmax_attention/dense/packed/*` + `plain_and_packed.cpp` (`a58a946c`) | same | — | already in tree: blobs byte-identical to sergiuszm (`kernel.cuh` 17089 B); test already registered |
| `include/ninfer/types.h` (`KvCacheStorage`) + `src/core/paged_kv_cache.{h,cpp}` | same | +85/+101 | P2 — RK/E8 enum values + storage modes; P1 forbidden from touching `types.h` |
| `kernel/gqa_attention_{decode,prefill}_*.cuh` (5 files) + launcher/wrapper | `causal_cache/{small_t,prompt}_{bf16,i8}.cuh` + `{small_t,prompt}.cu` | 887 | re-align: `E8Lattice/E8Root` branches grafted into v3 kernels, dispatch follows v3 layout |
| `src/targets/qwen3_6/**` | `src/models/qwen3_5/**` | varies | re-align: v3 renamed target→model, semantics 1:1 |
| `apps/cli/**`, `src/serve/**`, `src/runtime/engine/**`, `tests/test_kv_cache.cpp` | same | varies | P3+/re-align — CLI·serve·engine wiring out of P1 |

**YaRN stream** (`93419f4d`, 48 files +1125/−249): rope h/cuh/cu/wrapper, `contract/yarn.h` (A), yarn
tests, `verify_yarn_context.py`, `generate_yarn_fixture.py` = drop-in; `types.h` P2; `targets/qwen3_6`→
`models/qwen3_5` re-align. **Second stream** (`FORK..aeeba414`, 1063 files): #9 deadlock fix
`539ccdcd`+`2c046095`+`81b68a20` (branch `fix/issue9-entitlement`@`81b68a20`, priority), #6 `01c22ab6`,
#8 `29999279`, per-item vision cap `328d9aa8`, packed consolidation `a58a946c` (already absorbed by
v3), DFlash2/MoE/nvfp4 tuning out of scope.

## ② v3 rename/delete conflict hotspots

- `gqa_attention_*` split into `softmax_attention/dense/causal_cache/{small_t,prompt}_*` by the packed
  consolidation: upstream E8 branches sit on pre-packed names → branch-level graft only.
- `targets/qwen3_6/**`→`models/qwen3_5/**`; `export/ninfer/targets/qwen3_6/*` headers have no v3
  counterpart — verify `include/ninfer/` equivalents before P2 wiring.
- `types.h` restructured in v3 (`KvCacheStorage` now 5 values): upstream +85 lines are entangled with
  engine/serve fields; P2 takes only enum values + storage dispatch.

## ③ Fallback

If xkeyc's 6 commits (`69e6ae19`…`639e926a` + merge `93419f4d`) diverge >~200 lines semantically from
v3 attention kernels: rebase onto sergiuszm tip `aeeba414` and cherry-pick xkeyc's 6 commits. E8 codec
blobs are identical on both bases, so a base switch only re-aligns YaRN/prefix-cache.
