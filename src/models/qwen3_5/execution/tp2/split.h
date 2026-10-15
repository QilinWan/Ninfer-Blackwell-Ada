#pragma once

// The two-device (tp == 2) execution forms in one include, for the Program and the planner.
// Our file: the upstream execution headers carry no hook.

#include "models/qwen3_5/execution/tp2/attention_split.h"
#include "models/qwen3_5/execution/tp2/draft_split.h"
#include "models/qwen3_5/execution/tp2/ffn_split.h"
#include "models/qwen3_5/execution/tp2/gdn_split.h"
#include "models/qwen3_5/execution/tp2/linear_split.h"
#include "models/qwen3_5/execution/tp2/mtp_split.h"
#include "models/qwen3_5/execution/tp2/workspace_split.h"
#include "models/qwen3_5/execution/tp.h"
#include "core/tp2/device_tuning.h"
