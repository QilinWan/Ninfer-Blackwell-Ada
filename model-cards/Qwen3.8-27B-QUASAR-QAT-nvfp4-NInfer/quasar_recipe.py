"""QUASAR-QAT Qwen3.8-27B NVFP4 -> ninfer (spike 24/09).
Every projection that QUASAR stores as NVFP4 is imported as-is (codes, block scales and global scale, no
requantization); projections QUASAR keeps in BF16 (lm_head, and any other ignored linear) become FP8 rows,
the official qwen3_8_27b_nvfp4 method for the embedding. Vision/MTP follow the official _optional choices."""
from tools.convert.methods import fp8_row_maxabs, import_encoded
from tools.convert.official_recipes import FP8, _optional

REPORT = {"nvfp4": [], "fp8_from_bf16": [], "as_is": []}


def configure(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        if not parameter.projection or name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.assign(name, source=model.source(name, quantized))
            REPORT["as_is"].append(name)
            continue
        if name == "text/output_head":  # QUASAR keeps lm_head in BF16 (quantization ignore list)
            recipe.assign(name, format=FP8, method=fp8_row_maxabs, source=model.source(name, quantized),
                          activation_policy="AllowA8")
            REPORT["fp8_from_bf16"].append(name)
            continue
        recipe.assign(name, format="nvfp4", method=import_encoded,
                      source=model.source(name, quantized, "nvfp4"), activation_policy="AllowA4")
        REPORT["nvfp4"].append(name)
    print("QUASAR recipe:", {k: len(v) for k, v in REPORT.items()}, "fp8_from_bf16:", REPORT["fp8_from_bf16"][:8], flush=True)
