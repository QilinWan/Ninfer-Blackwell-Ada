#pragma once

// What the two-device (tp 2) parts of the Qwen3.5 Program need, in one include for
// program/program_impl.h. Our file.

#include "core/device.h"
#include "core/device_scope.h"
#include "models/qwen3_5/execution/tp.h"
#include "models/qwen3_5/execution/tp2/draft_split.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/peer_mailbox.h"

#include <array>
