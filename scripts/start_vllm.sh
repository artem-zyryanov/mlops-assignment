#!/usr/bin/env bash
#
# Start vLLM with the assignment's chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Flag rationale (longer form in REPORT.md §1):
#  --tensor-parallel-size 1          single H100, no sharding
#  --max-model-len 8192              schema (1-3K) + question + headroom; smaller
#                                    window -> bigger KV cache budget per slot
#  --gpu-memory-utilization 0.90     leave 10% for CUDA scratch
#  --enable-prefix-caching           schema text is byte-identical across
#                                    requests for the same db_id; cache the prefill
#  --enable-chunked-prefill          smooth P95 by letting prefill share GPU with
#                                    decode under continuous batching
#  --max-num-seqs 64                 starting concurrency cap (tuned in Phase 6)
#  --max-num-batched-tokens 8192     per-step token budget (tuned in Phase 6)
#  --dtype bfloat16                  H100-native

set -euo pipefail

# Non-interactive ssh skips ~/.profile; make sure uv is reachable.
export PATH="$HOME/.local/bin:$PATH"

MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --disable-log-requests \
    --uvicorn-log-level warning
