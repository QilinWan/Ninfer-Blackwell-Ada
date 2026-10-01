"""Move only the output head of the 9B NVFP4 recipe to weight-only NVFP4.

Measured against qwen3_5_9b_nvfp4_head_mtp.py: quantizing the MTP layer too shortened the
MTP3 round by ~5% but cost the same in acceptance (3.16 against 3.32 tokens per round), a
wash; the head alone keeps the draft at Q8. Served at A16 (no divisor for text/final_hidden).
"""

from __future__ import annotations

from tools.convert.methods import nvfp4_blockwise


def configure(model, recipe, sources) -> None:
    recipe.assign("text/output_head", format="nvfp4", method=nvfp4_blockwise)
    print("NVFP4 head override applied")
