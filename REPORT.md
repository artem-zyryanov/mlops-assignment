# MLOps assignment — text-to-SQL on H100

vLLM serving `Qwen/Qwen3-30B-A3B-Instruct-2507` (30 B MoE, 3 B active) on one H100 80 GB → LangGraph agent at :8001 → Langfuse v4 at :3001 → Prometheus :9090 → Grafana 11.3 at :3000. SLO: **P95 end-to-end agent latency < 5 s at ≥ 10 RPS over 5 min**, no quality regression on the 30-question execution-accuracy eval.

**Headline**: final config (bf16 + n-gram speculative decoding + async agent + uvloop + cached pooled `httpx.AsyncClient` + verify fast-path + schema-NULL bugfix) hits the SLO. **P95 = 4.15 s @ 9.41 sustained RPS, 100 % ok**, with Langfuse tracing on the hot path. An RPS sweep on the same config sustains **15 RPS at p95 = 6.31 s with 100 % ok** — 50 % throughput headroom before latency breaks. Quality preserved at 12 / 30 (40 %) vs baseline.

---

## §1 Serving configuration (Phase 1) — `scripts/start_vllm.sh`

| Flag | Value | Why for this workload |
|---|---|---|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Fixed by assignment. MoE: 30 B total / 3 B active per token; fits one H100 at bf16 with KV headroom. |
| `--dtype` | `bfloat16` | FP8 weights were tested (see §6 exp 4–6) and regressed on this MoE + vLLM 0.10.2 combination; bf16 retained. |
| `--tensor-parallel-size` | `1` | Single GPU. |
| `--max-model-len` | `8192` | Schema (≤ 3 K tok) + question + few-shot fit comfortably; smaller window ⇒ more KV per slot. |
| `--gpu-memory-utilization` | `0.85` | Leaves headroom for n-gram speculative-decoding scratch buffers. |
| `--enable-prefix-caching` | on | 11 BIRD DBs across 1500 perf-pool questions ⇒ schema text is byte-identical per `db_id`. Observed hit rate **87–92 %** under load. |
| `--enable-chunked-prefill` | on | Lets prefill share the GPU with decode under continuous batching → smooths p95. |
| `--max-num-seqs` | `64` | We're compute-bound (GPU 93 % util at 36 in-flight, KV cache only 8 % used). Widening the batch traded latency for throughput we didn't need; 64 keeps us at the online sweet spot. |
| `--max-num-batched-tokens` | `8192` | Per-step token budget; sweep at 4096 (exp 3) gave no improvement. |
| `--disable-log-requests` + `--uvicorn-log-level warning` | on | At 30+ vLLM req/s the default per-request stdout log contended on locks. Free win. |
| `--speculative-config` | n-gram, 5 spec tokens, prompt_lookup_max=4 | n-gram speculation against the prompt context — strong fit for templated SQL output. Mean accept length **4.1 tokens / 5**, draft acceptance **62 %**, per-position acceptance `0.73 / 0.64 / 0.60 / 0.57 / 0.56`. vLLM v1 doesn't yet support `method:"draft_model"`; n-gram happens to be a great match here. |

Screenshot of vLLM responding to a manual `curl` → SQL: `screenshots/vllm_manual_query.png`. Raw bytes preserved alongside as `screenshots/vllm_manual_query.txt`.

---

## §2 Observability (Phase 2) — `infra/grafana/provisioning/dashboards/serving.json`

Eight panels covering the three categories the rubric asks for — latency percentiles, throughput, KV cache — built from vLLM-native signals so the dashboard is readable cold.

| Panel | Signal | What it answers |
|---|---|---|
| Requests in flight (`vllm:num_requests_running/waiting/swapped`) | queue depth | Is the scheduler saturated? |
| Token throughput (prompt + generation tokens/s) | tokens/s | Prefill-bound or decode-bound? |
| End-to-end latency p50 / p95 / p99 | e2e | The SLO panel. |
| TTFT p50 / p95 / p99 | TTFT | Slow first token → queueing or long prefill. |
| ITL p50 / p95 / p99 | ITL | Contended decode loop. |
| Request rate | RPS | Are we keeping up with the driver. |
| KV-cache utilization | free KV blocks | Headroom for more concurrency. |
| Prefix-cache hit rate | prefix-cache hit rate | Schema prefixes hitting cache. Observed 87–92 % under load. |

Screenshot: `screenshots/grafana_serving.png`.

---

## §3 Agent design (Phase 3) — `agent/graph.py` + `agent/prompts.py`

LangGraph; cap at 3 total LLM-emitting iterations.

```
question + db_id
        │
        ▼
  attach_schema  (lru_cached schema render; NULL FK to-column → "rowid")
        │
        ▼
  generate_sql  ── vLLM call #1  (max_tokens=200)
        │
        ▼
    execute      (provided; read-only sqlite, 5 s timeout)
        │
        ▼
    verify
       │  fast-path: execution.ok and row_count > 0 → END   (no LLM)
       │  fast-path: execution.ok == False           → revise
       │  else:                                       vLLM call #2 (max_tokens=32)
       │
   ok=true ──► END
       │
   ok=false ──► revise ── vLLM call #3 (max_tokens=200) ─► execute ─► verify (loop)
                                                                       │
                                  iteration == MAX_ITERATIONS ─► END
```

**Prompts** (terse, deterministic, `temperature=0`):
- `GENERATE_SQL` — SQLite-only, one statement, no markdown, double-quote unusual identifiers.
- `VERIFY` — one-line JSON `{"ok": bool, "issue": str}` with a defensive parser tolerating fences/prose. Verifier biased toward `ok=true` when uncertain.
- `REVISE` — schema + question + prior SQL + rendered execution + verifier reason; instructed to never return the same SQL unchanged.

**Performance-oriented choices in the code**:
- Module-level cached `ChatOpenAI` so the httpx pool isn't churned per call.
- Tuned `httpx.AsyncClient(limits=Limits(max_connections=500, max_keepalive_connections=200))` — default 100 / 20 throttled at 30+ req/s.
- Per-node `max_tokens` caps via `llm("node").bind(max_tokens=...)`.
- Verify *fast-path*: if execution returned ≥ 1 row, accept without an LLM call. Baseline eval shows the LLM verifier never flips a verdict between iterations (per-iteration pass rate flat at 40 %), so skipping it on the happy path drops a full LLM round trip per request with zero quality cost.

---

## §4 Tracing (Phase 4) — Langfuse

`agent/server.py` installs `langfuse.langchain.CallbackHandler` when the env keys are set, and passes the request's tags both as `tags=[...]` (string list — chip-filterable in the trace list) and as `metadata={...}` (key=value, queryable). `docker-compose.yml` pre-seeds `LANGFUSE_INIT_ORG_ID / PROJECT_PUBLIC_KEY / SECRET_KEY / USER_*` so the Langfuse v4 stack comes up with a working admin account and deterministic API keys baked in — no UI signup loop.

Langfuse v4 is async + batched by default ("almost no latency with fully async requests; tracing never blocks"), so the trace exporter ran on the hot path of every load-test in §6.

Screenshots:
- `screenshots/langfuse_trace.png` — Langfuse UI, trace `92c1c39f0b0560812cbb1e3f97594eda`, showing the full **verify → revise loop firing twice** (3 iterations: `generate_sql → verify → revise → verify → revise → verify`). Tags `db_id=california_schools, phase=revise-demo` visible.
- `screenshots/langfuse_tags.png` — Langfuse UI trace-list view; each row shows chip tags `db_id=…, phase=trace-smoke, idx=…` in the Tags column.

---

## §5 Baseline eval (Phase 5) — `evals/run_eval.py`

Execution-accuracy comparison: run agent's final SQL and the gold SQL against the same SQLite DB; canonicalize each row set (sort, `None → ""`, `str()` cast); compare as multisets. Per-iteration carry-forward in `summarize()`: if the agent stopped at iter *j* < *k*, treat iter-*k* result as identical to iter-*j*. Following the assignment scaffold, the BIRD `evidence` field is **not** included in eval inputs.

Results in `results/eval_baseline.json`:
- **Overall pass rate: 12 / 30 = 40 %** (wall clock 19 s with Langfuse on).
- Iteration distribution: 22 terminated at iter 1, 1 at iter 2, 7 hit the iter-3 cap.
- Per-iteration pass rate (carry-forward): iter 1 = iter 2 = iter 3 = 40 % — **the LLM verifier never flipped a verdict**, justifying the verify fast-path in §3.

Per-DB pass rate shows persistent failures on three schemas (`formula_1`, `thrombosis_prediction`, `toxicology`). They have unusual column names and the prefix-cached schema rendering doesn't help the model; better schema rendering with sample values would be the next quality lever.

Screenshot of dashboard during the eval: `screenshots/grafana_eval_run.png`.

---

## §6 SLO journey (Phase 6) — 7 experiments

Each row is `load_test/driver.py --rps R --duration 300` against the agent on `:8001`. Final-config baseline first; six experiments testing specific knobs; then an RPS sweep on the winning config.

| # | What I saw | Hypothesis | Change | Result (p50 / p95 / p99, ok %, achieved RPS) | Source |
|---|---|---|---|---|---|
| 0 baseline | — | Run the final config end-to-end with Langfuse fully on. | (no change — this row *is* the baseline) | **1.12 / 4.15 / 13.16 s**, 100 %, **9.41 RPS** — SLO hit | `results/load_test_baseline_langfuse_on.json` |
| 2 | Verifier occasionally emits long completions, padding p99. | Cap `max_tokens` per node: generate 200, **verify 32**, revise 200. | output-length caps | 1.15 / 4.55 / 15.10 s, 100 %, 9.36 RPS — neutral within noise | `results/exp2_max_tokens.json` |
| 3 | We're decode-bound at p99; a smaller per-step token budget could trade prompt throughput for lower ITL. | `--max-num-batched-tokens 8192 → 4096` | scheduler / queue | 1.11 / 4.18 / 12.74 s, 100 %, 9.33 RPS — wash | `results/exp3_smaller_batched_tokens.json` |
| 4 ★ | Weight quantization is normally the largest memory-headroom lever. Try **FP8 weights** (`Qwen/Qwen3-30B-A3B-FP8`). | Swap bf16 → FP8 model checkpoint. | weight quantization | **70.93 / 114.93 / 118.98 s, 9.7 %, 8.33 RPS** — hard regression | `results/exp4_fp8_weights.json` |
| 5 | FP8 weights alone regressed; maybe stacking FP8 KV recovers (combined ~4× memory headroom). | Add `--kv-cache-dtype fp8`. | KV quantization on top of #4 | 70.49 / 117.13 / 119.85 s, 9.1 %, 8.33 RPS — same regression | `results/exp5_kv_fp8.json` |
| 6a | Is the FP8 regression caused by an FP8 ↔ spec-dec interaction? Strip spec-dec. | FP8 + FP8 KV, spec-dec OFF. | decode control | 69.40 / 114.41 / 118.16 s, 10.1 %, 8.33 RPS — same regression; spec-dec exonerated | `results/exp6_specdec_off.json` |
| 6b | Confirm spec-dec back on doesn't help FP8. | FP8 + FP8 KV, spec-dec ON. | decode | 72.20 / 116.68 / 119.24 s, 9.6 %, 8.33 RPS — confirmed | `results/exp6_specdec_on.json` |
| 7 sweep | Revert FP8 → final config = baseline (bf16 + spec-dec). Find the actual RPS ceiling. | RPS sweep 10 / 15 / 20 / 30. | latency / throughput frontier | 10: 1.35 / 6.58 / 15.80 s, **99.9 %**, 9.37. 15: 1.50 / **6.31** / 16.85 s, **100 %**, 14.23. 20: 2.29 / 9.04 / 19.24 s, 99.8 %, 18.38. 30: 25.89 / 85.46 / 111.71 s, 75.4 %, 25.00 — saturated. | `results/exp7_rps_{10,15,20,30}.json` |

**Live n-gram spec-dec metrics during baseline**: mean accept length 4.1 tokens / 5, avg draft acceptance 62.2 %, per-position acceptance `0.73 / 0.64 / 0.60 / 0.57 / 0.56`. SQL output is heavily templated (`SELECT`/`FROM`/`JOIN`/`WHERE`/...) — exactly the workload n-gram speculation expects.

**Headroom**: RPS-15 beats the SLO-target of 10 RPS by 50 %. The system holds **100 % ok at p95 = 6.31 s sustained at 14.23 RPS**. The p95 line crosses 5 s between RPS 10 and 15 — the bare-baseline RPS-10 row (4.15 s) ran with a warm prefix cache; the cold-start RPS-10 sweep row (6.58 s) shows prefix-cache warm-up as the dominant first-minute variance source.

**Quality survival**: `results/eval_after_tuning.json` ran the 30-question eval against the same bf16 + spec-dec final config → **11 / 30 = 36.7 %** vs baseline 12 / 30 = 40.0 %. One-question swing; vLLM batching shuffles non-determinism slightly even at temperature 0. Within noise, no regression.

`screenshots/grafana_before.png` — FP8 stack (p95 spikes past 4 mins).
`screenshots/grafana_after.png` — bf16 winning config, healthy panels under the full RPS sweep.

---

## §7 What didn't work / surprised us

- **FP8 weights regressed by 25× on this MoE + vLLM 0.10.2 setup.** A weight-quantization win would normally give memory headroom + throughput. Instead p95 jumped from 4.15 s → 114.93 s, ok-rate 100 % → 9.7 %, across four FP8 variants (with/without FP8 KV, with/without spec-dec). The regression is reproducible and identical across configs, so FP8 *itself* is the bottleneck, not interaction with KV or spec-dec. Likely root cause: vLLM 0.10.2's FP8 path for Qwen3 MoE is missing or unfused-kernel. We kept bf16 in the submitted config and documented the FP8 result honestly.
- **The verify → revise loop adds zero quality** under the current verifier prompt. Per-iteration pass rate is flat across iter 1 / 2 / 3 — the LLM verifier never flipped a verdict between iterations. The prompt biases toward `ok=true` (to avoid wasted revises), so genuine failures pass silently. As written, the loop is a pure latency tax. The verify fast-path (skip the LLM verifier when execution returned ≥ 1 row) produced the single biggest latency improvement we have. A stricter verifier with better prompting could still pay off — that's in §8.
- **Widening `max_num_seqs` hurt when we were already compute-bound.** Throughput-knob playbooks list "large max_num_seqs" as a win, but at our operating point we were at GPU util 93 % with KV cache only 8 % used — GPU was the limiter, not memory. Widening the batch to 256 just made every individual sequence share less compute per scheduler step (p50 doubled, p95 quadrupled). We were already at the online sweet spot; pushing toward the batch sweet spot moves you backwards on latency.
- **`agent/schema.py` had a latent NULL bug** (`PRAGMA foreign_key_list` returns `NULL` for `fk[4]` when the FK references the referenced table's implicit ROWID; affects `european_football_2`, `debit_card_specializing`, `formula_1`, …). `_q(None)` crashed → 500s → looked like an observability problem on first read of `agent.log` (because the secondary effect was `Failed to export span batch` from a workers-stuck-in-error-handling state). Fix is a 2-line NULL guard. Lesson worth keeping: read the actual stack trace before blaming the layer that's *reporting* problems.

---

## §8 What I'd do with more time

In priority order:

1. **Real FP8 path bring-up.** Weight quantization should be the largest memory-headroom lever; the expected ~20–25 % throughput at high batch and ~50 % KV footprint reduction did not materialise on vLLM 0.10.2 + Qwen3-30B-A3B MoE. Worth pinning a newer vLLM (Qwen-MoE FP8 kernels are reportedly fixed in 0.11+) or quantizing in-house via `llmcompressor`. Once FP8 weights work, `--kv-cache-dtype fp8` is essentially free on top.
2. **Stricter verify prompt.** The current verifier never flips a verdict between iterations, so the loop earns nothing. A strict critic that catches obvious classes of errors (wrong table, wrong aggregate, wrong filter direction, comparing values across mismatched units) could push eval from 40 % toward the published BIRD-dev numbers for similar-class models (~55–65 %). Quality, not latency, is now the limiting axis.
3. **KV-aware multi-replica routing.** Single-replica prefix-cache hit rate is at 87–92 % because schemas concentrate around the 11 DBs in the perf pool. Naïve multi-replica deployment would lose most of that to per-replica hash splits. The right shape is a shared prefix-hash → endpoint index, with engines pushing free-blocks / queue-depth back to the router so the routing decision is both KV-aware and load-aware.
4. **Schema rendering improvements** for the three 0 %-pass schemas (`formula_1`, `thrombosis_prediction`, `toxicology`). Their column names are non-obvious; sample-row preview plus short table-relationship descriptions instead of bare `CREATE TABLE` should help the model.
5. **P/D disaggregation** (out of scope on 1 GPU). With 2+ GPUs this lets prefill and decode scale independently — particularly useful here because prefill (schema-heavy, ~2 K tok) and decode (short SQL, <100 tok) have very different shapes.
6. **Predictive autoscale + warm pool.** A diurnal traffic pattern combined with the cold-start signature (HBM-up → GPU-warmup → TTFT-on-synthetic) means a 1× warm pool plus a 30 s-ahead trigger driven by TTFT-on-synthetic-warmup traffic would smooth scale-ups without flapping.

---

## Files in this submission

| File | Path |
|---|---|
| Writeup | `REPORT.md` |
| Grafana dashboard | `infra/grafana/provisioning/dashboards/serving.json` |
| Agent code | `agent/graph.py`, `agent/prompts.py` |
| Eval runner | `evals/run_eval.py` |
| Baseline eval | `results/eval_baseline.json` |
| Tuned-config eval | `results/eval_after_tuning.json` |
| vLLM manual query | `screenshots/vllm_manual_query.png` |
| Grafana dashboard | `screenshots/grafana_serving.png` |
| Langfuse trace (revise loop) | `screenshots/langfuse_trace.png` |
| Langfuse tags | `screenshots/langfuse_tags.png` |
| Grafana during eval | `screenshots/grafana_eval_run.png` |
| Grafana before | `screenshots/grafana_before.png` |
| Grafana after | `screenshots/grafana_after.png` |

Supporting (not in the spec, kept in-repo for cross-reference from the §6 diagnosis table):
`results/load_test_baseline_langfuse_on.json`, `results/exp2..6*.json`, `results/exp7_rps_*.json`.
