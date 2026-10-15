"""Generate independent Qwen YaRN coefficients with Hugging Face Transformers on CPU.

Use an existing local Qwen3.8-27B config. This never loads model weights or downloads files.
"""
import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig
from transformers.modeling_rope_utils import _compute_yarn_parameters

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
config = AutoConfig.from_pretrained(args.config, local_files_only=True).get_text_config()
parameters = {
    "rope_type": "yarn", "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25, "mrope_interleaved": True,
    "mrope_section": [11, 11, 10], "original_max_position_embeddings": 262144,
    "beta_fast": 32, "beta_slow": 1,
}
rows = []
for factor in (1.0, 1.5, 2.0, 4.0):
    config.rope_parameters = dict(parameters, factor=factor)
    inv_freq, attention_factor = _compute_yarn_parameters(config, torch.device("cpu"))
    rows.append({"factor": factor, "inverse_frequency": inv_freq.tolist(),
                 "attention_factor": attention_factor})
Path(args.output).write_text(json.dumps({
    "source": "transformers.modeling_rope_utils._compute_yarn_parameters",
    "transformers_version": transformers.__version__,
    "torch_version": torch.__version__, "parameters": parameters,
    "head_dim": config.head_dim, "cases": rows,
}, indent=2) + "\n")
print("Generated CPU YaRN reference with Transformers", transformers.__version__)
