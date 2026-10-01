"""Grant AllowA4 with calibrated activation divisors to every NVFP4 site of the 9B recipe.

Use as `--override tools/convert/overrides/qwen3_5_9b_nvfp4_a4.py` after
`--recipe qwen3_5_9b_nvfp4`, with NINFER_ACTIVATION_AMAX naming the JSON written by
tools/calibrate_activation_amax.py. Every input site gets d_x = 448 * 6 / amax: the
largest calibrated activation lands exactly on the E4M3FN maximum, so nothing seen in
calibration saturates (nvfp4_codec.cuh quantize_nvfp4_k16). Parents that group several
parameters (GDN query|key|value, MLP gate|up) read one input site, so their divisors are
bit-identical, which ops/weight_input.cpp requires. Sites the calibration does not name stay
A16Only rather than guess.
"""

from __future__ import annotations

import json
import os
import struct

E2M1_MAX = 6.0
E4M3FN_MAX = 448.0


def configure(model, recipe, sources) -> None:
    path = os.environ.get("NINFER_ACTIVATION_AMAX")
    if not path:
        raise ValueError("set NINFER_ACTIVATION_AMAX to the calibration JSON path")
    sites = json.load(open(path))["sites"]
    granted = 0
    for name, selections in recipe.selections.items():
        if not any(s.format == "nvfp4" for s in selections):
            continue
        for site in model.parameters[name].inputs:
            amax = sites.get(site)
            if amax is None:
                continue
            divisor = struct.unpack("<f", struct.pack("<f", E4M3FN_MAX * E2M1_MAX / float(amax)))[0]
            recipe.use(
                name,
                site,
                activation_policy="AllowA4",
                auxiliaries={"activation_input_divisor": divisor},
            )
            granted += 1
    if granted == 0:
        raise ValueError("calibration named no NVFP4 input site of this model")
    print(f"AllowA4 granted at {granted} NVFP4 uses from {path}")
