"""The opt-in draft-head helpers of the official recipes store exactly the projections they name."""

from __future__ import annotations

import torch

from tools.convert.methods import grouped_absmax, nvfp4_mse
from tools.convert.official_recipes import (
    _dflash2_nvfp4_gate_up,
    _mtp_nvfp4_layer,
    _optional,
)
from tools.convert.qwen3_5 import build_model
from tools.convert.recipe import Recipe

from .test_qwen3_5 import _checkpoint, _config

DRAFT = {
    "architectures": ["DFlash2DraftModel"],
    "hidden_size": 16,
    "vocab_size": 8,
    "num_hidden_layers": 2,
    "intermediate_size": 24,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "max_position_embeddings": 128,
    "layer_types": ["sliding_attention", "sliding_attention"],
    "sliding_window": 32,
    "dflash_config": {
        "target_layer_ids": [1, 0],
        "mask_token_id": 7,
        "conv_kernel_size": 2,
        "conv_group_size": 4,
        "selector_rank": 4,
        "selector_top_k": 2,
    },
}


def _choice(recipe, name):
    (selection,) = recipe.selections[name]
    return selection.format, selection.method


def _policies(recipe, name):
    return {policy for (parameter, _), policy in recipe.policies.items() if parameter == name}


def _model(tmp_path, components):
    base = _checkpoint(tmp_path / "text", _config(), {"unused": torch.ones(1)})
    companions = {}
    if "dflash2" in components:
        companions["dflash2"] = _checkpoint(tmp_path / "draft", DRAFT, {"unused": torch.ones(1)})
    return build_model(base, components=components, companions=companions)


def test_mtp_nvfp4_layer_converts_mlp_and_attention_output_only(tmp_path):
    model = _model(tmp_path, ("text", "mtp"))
    recipe = Recipe(model)
    _optional(model, recipe)
    _mtp_nvfp4_layer(model, recipe)
    for name in (
        "mtp/layers/0/mlp/gate",
        "mtp/layers/0/mlp/up",
        "mtp/layers/0/mlp/down",
        "mtp/layers/0/attention/output",
    ):
        assert _choice(recipe, name) == ("nvfp4", nvfp4_mse), name
        assert _policies(recipe, name) == {"A16Only"}, name
    for name in (
        "mtp/input_projection",
        "mtp/layers/0/attention/query",
        "mtp/layers/0/attention/key",
        "mtp/layers/0/attention/gate",
        "mtp/layers/0/attention/value",
    ):
        assert _choice(recipe, name) == ("q8_g32_fp16", grouped_absmax), name


def test_mtp_nvfp4_layer_without_mtp_changes_nothing(tmp_path):
    model = _model(tmp_path, ("text",))
    recipe = Recipe(model)
    before = {name: list(choices) for name, choices in recipe.selections.items()}
    _mtp_nvfp4_layer(model, recipe)
    assert recipe.selections == before


def test_dflash2_nvfp4_gate_up_leaves_the_rest_of_the_drafter_q8(tmp_path):
    model = _model(tmp_path, ("text", "dflash2"))
    recipe = Recipe(model)
    _optional(model, recipe)
    _dflash2_nvfp4_gate_up(model, recipe)
    for layer in range(2):
        prefix = f"dflash2/layers/{layer}/"
        for leaf in ("mlp/gate", "mlp/up"):
            assert _choice(recipe, prefix + leaf) == ("nvfp4", nvfp4_mse), prefix + leaf
            assert _policies(recipe, prefix + leaf) == {"A16Only"}, prefix + leaf
        # The single-device drafter fuses the down and attention output projections with the
        # dynamic convolution's finish (linear_dynamic_grouped_conv_add), which takes Q8 only.
        for leaf in ("mlp/down", "attention/query", "attention/output"):
            assert _choice(recipe, prefix + leaf) == ("q8_g32_fp16", grouped_absmax), prefix + leaf
