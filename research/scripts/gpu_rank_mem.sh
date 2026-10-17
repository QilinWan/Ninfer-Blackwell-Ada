#!/usr/bin/env bash
# gpu_rank_mem.sh - snapshot per-GPU used memory and per-process (per-rank) used memory.
# Usage: bash gpu_rank_mem.sh [tag]
tag=${1:-snap}
echo "== $tag $(date -u +%H:%M:%S)"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "-- compute apps (pid, gpu_uuid, used_memory)"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
