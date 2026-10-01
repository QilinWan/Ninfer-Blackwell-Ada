"""Move the output head and the MTP stem/MLP of the 9B NVFP4 recipe to weight-only NVFP4.

Use after `--recipe qwen3_5_9b_nvfp4` (and after the A4 override if both are wanted; this
one only reassigns formats). The decode profile of the A16/A4 artifact put the Q6 output head
at ~10% of an MTP3 round and the Q8 MTP layer at ~13%; these are the two largest of those
parents that the engine can read as whole NVFP4 parents. The MTP attention stays Q8: mtp.cpp
reads it as one packed q|k|gate|v parent *and* as four row views, and NVFP4 row views are
rejected. The head is served at A16 (no divisor is calibrated for text/final_hidden), which
only touches ninfer-perplexity's all-token scoring, not decode.
"""

from __future__ import annotations

from tools.convert.methods import nvfp4_blockwise


def configure(model, recipe, sources) -> None:
    for name in (
        "text/output_head",
        "mtp/input_projection",
        "mtp/layers/0/mlp/gate",
        "mtp/layers/0/mlp/up",
        "mtp/layers/0/mlp/down",
    ):
        if name not in recipe.selections:
            raise ValueError(f"{name}: not a parameter of this model")
        recipe.assign(name, format="nvfp4", method=nvfp4_blockwise)
    recipe.group(["mtp/layers/0/mlp/gate", "mtp/layers/0/mlp/up"])
    print("NVFP4 head + MTP stem/MLP override applied")
