# S11 — multimodal prefill at tp 2 (mechanism and limits)

Status: implemented; measured numbers below are filled from the runs listed in each section.

## What changed

`TextContext::prefill_impl_tp2` now runs a multimodal chunk:

1. rank 0's `VisionPrefillSession::prepare_chunk` encodes the active item into the handoff region
   of rank 0's arena (above `general_capacity`, as on one device);
2. rank 0 copies the chunk's `[hidden, visual_rows]` columns to rank 1's own arena with one
   `cudaMemcpy2DAsync` over UVA, ordered by a `PeerEvents::inputs_ready(0)` event and followed by
   rank 1's scatter on its own stream;
3. both ranks scatter the same rows into their replicated residual; the three M-RoPE axes come
   from the prepared prompt on both ranks (`[3,T]` axis-major, the same layout the rope kernel
   indexes);
4. the MTP prompt alignment builds rank 0's composed embedding (shifted ids + shifted visual rows)
   and the split MTP stem substitutes it for rank 0's token embedding; rank 1 still contracts the
   final hidden. The same substitution serves the MTP bridge's composed embedding.

`planning/startup.cpp` sizes both ranks' arenas for the receiver block (`[hidden, chunk]`), the
generic scatter vector and, with MTP, rank 0's composed embedding.

## Limits

- A chunk is capped at the active item's end by the Vision plan, so `--prefill-chunk` never splits
  an item across two encodes.
- The tower stays SingleDevice on rank 0 (`--vision-device` must be the first `--devices` entry);
  rank 1 never holds tower weights and never re-encodes.
- The MTP bridge still requires a one-axis text RoPE position: resuming a prefix whose frontier
  token is *inside* an item remains a single-device case. A text frontier after a media prompt
  works, and so does the composed embedding itself.
- DFlash (masked drafter) remains unsupported at tp 2, as before; its prefill-time round has no
  split form.

## Measurements

Hardware: 2x RTX 5060 Ti 16 GB (sm_120a, no P2P), CUDA 13.2, artifact
`qwen3_8_27b_quasar_nvfp4.ninfer` (17,397,699,076 B, sha256 `808a0fa3…b570`, text+vision+mtp).

| check | before (`b55da868`) | after |
|---|---|---|
| image prompt, `ninfer-serve --tp 2 --vision` | HTTP 500 `multimodal prefill is not implemented at tensor-parallel width 2` | **HTTP 200**, red image -> `Red`, blue image -> ` blue` |
| 60 greedy text prompts, tp 2 | baseline decode 46.0 tok/s, prefill 434.6 tok/s | token ids **60/60 identical** |
| same image twice, greedy | n/a | identical token ids |
| two images in one prompt | n/a | `red, blue`, two runs identical |
| 1024x512 image | n/a | ` green` |
| MTP (`--spec mtp --draft-tokens 3`) + image | n/a | HTTP 200, `mtp accepted 3/3 (100.0%)` |
| tp 2 ctest set (`ctest -R tp2`, this run) | n/a | 8 passed, 0 failed, 2 skipped (DFlash2 not in artifact) of 10 |

Commands and logs: `notes/W1-repro.md`, `W3-regression.md`, `W5-handoff.md`, `W7-multi.md`,
`W9-mtp-vision.md`, `W11-suites.md` (workspace `/data/s11/notes`).

## Pre-existing gaps this task did not change

- `EngineOptions::max_vision_tokens` is not consumed by this tree: a 768x768 image counts 576
  merged tokens whatever the option says (the same on `b55da868`). It sizes nothing; the Vision
  workspace follows the context-derived prompt budget. The tp 2 test now asserts the prompt
  budget instead of a per-item resize.
- `vision_device` placement is validated by the product layer (`product::validate_vision_options`,
  used by the CLI and `ninfer-serve`), not by `EngineOptions` construction. The Engine test dropped
  the checks that asserted constructor-time rejection; the CLI rejects all three cases before it
  loads the model.
- The artifact the plan recommends has no DFlash2 component, so the DFlash2 tp 2 suites report
  skip 77, not failure.
