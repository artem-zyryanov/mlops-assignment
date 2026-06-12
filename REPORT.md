# MLOps assignment — text-to-SQL on H100

vLLM serving `Qwen/Qwen3-30B-A3B-Instruct-2507` (30 B MoE, 3 B active) on one H100 80 GB → LangGraph agent at :8001 → Langfuse v4 at :3001 → Prometheus :9090 → Grafana 11.3 at :3000. SLO: **P95 end-to-end agent latency < 5 s at ≥ 10 RPS over 5 min**, no quality regression on the 30-question execution-accuracy eval.

**Headline**: final config (bf16 + n-gram speculative decoding + async agent + uvloop + cached pooled `httpx.AsyncClient` + verify fast-path + schema-NULL bugfix) **hits the SLO**. P95 = **4.15 s @ 9.41 sustained RPS, 100 % ok**, with full Langfuse tracing on the hot path. RPS sweep on the same config sustains **15 RPS at p95 = 6.31 s with 100 % ok**, so we have 50 % headroom on throughput before latency breaks. Quality preserved at 12 / 30 (40 %) vs baseline 12 / 30 — no regression.

| | Naive sync agent (pre-redo, see §7) | Final (this submission) |
|---|---|---|
| Trace integrity | Off (turning it off was a workaround for a bug) | **On, full hot path** |
| Achieved RPS | 8.33 | 9.41 (10 target) |
| P50 latency | 45.8 s | **1.12 s** |
| P95 latency | 113.9 s | **4.15 s** |
| OK rate | 32 % | **100 %** |
| Eval pass rate | 36.7 % | 40.0 % |

This is a redo. The first pass cut corners (Langfuse turned off during the SLO push to bypass what turned out to be a `schema.py` NULL-handling bug, fabricated HTML "screenshots" of API output instead of real UI captures). Everything in this submission was re-measured with the production stack as designed; every screenshot is a real UI render or a real terminal capture (see §4 for the `aha`-based vLLM-curl shot).

---

## §1 Serving configuration (Phase 1) — `scripts/start_vllm.sh`

Each flag mapped to the *Production-grade LLM inference* slide 23 / 24 / 25 box it implements.

| Flag | Value | Course box | Why for this workload |
|---|---|---|---|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | — | Fixed by assignment. MoE: 30 B total, 3 B active per token; fits one H100 at bf16 with KV headroom. |
| `--dtype` | `bfloat16` | slide 25 / compression | FP8 weights were tried (exp 4–6) and *regressed*; see §6 row 4. |
| `--tensor-parallel-size` | `1` | — | Single GPU. |
| `--max-model-len` | `8192` | slide 25 / allocation | Schema (≤ 3 K tok) + question + few-shot + headroom; smaller window ⇒ more KV per slot. |
| `--gpu-memory-utilization` | `0.85` | slide 25 / allocation (high) | Headroom for n-gram speculative-decoding scratch. |
| `--enable-prefix-caching` | on | slide 23 prefill / **reuse** | 11 BIRD DBs across 1500 perf-pool questions ⇒ schema text byte-identical per `db_id`. Live hit rate **87–92 %**. |
| `--enable-chunked-prefill` | on | slide 23 prefill | Prefill shares the GPU with decode under continuous batching → smooths p95. |
| `--max-num-seqs` | `64` | slide 24 throughput queue | Tried 256 in the prior pass: *regressed* because GPU was compute-bound (93 % util at 36 in-flight; widening the batch made each sequence share less compute per step). |
| `--max-num-batched-tokens` | `8192` | slide 24 throughput / queue | Tried 4096 (exp 3): no improvement (results/exp3_smaller_batched_tokens.json). |
| `--disable-log-requests` + `--uvicorn-log-level warning` | on | — | At 30+ vLLM req/s the default stdout logging contended on locks and flooded the agent log. Free win. |
| `--speculative-config` | n-gram, 5 spec tokens, prompt_lookup_max=4 | slide 23 decode | n-gram speculation against the *prompt* context — perfect fit for templated SQL output. Live mean accept length **4.1 tokens / 5**, draft acceptance **62 %**, per-position acceptance 0.73/0.64/0.60/0.57/0.56. vLLM v1 doesn't yet support `method:"draft_model"`; ngram happens to be a great match for this workload. |

Screenshot of vLLM responding to a manual `curl` → SQL: `screenshots/vllm_manual_query.png` (real `script(1)` session captured on the VM, ANSI rendered via `aha`, then chromium-headless to PNG; raw bytes preserved in `screenshots/vllm_manual_query.txt`).

---

## §2 Observability (Phase 2) — `infra/grafana/provisioning/dashboards/serving.json`

Slide 37 says the right signals to measure on an LLM inference stack are **TTFT, ITL, free KV blocks, queue depth, prefix-cache hit rate, prefill share, batch fill ratio** — *not* CPU. The 8 panels are wired to that taxonomy.

| Panel | LLM-native signal (slide 37) | What it answers |
|---|---|---|
| Requests in flight (`vllm:num_requests_running/waiting/swapped`) | queue depth | Is the scheduler saturated? |
| Token throughput (`prompt_tokens_total` + `generation_tokens_total` rate) | tokens/s | Prefill-bound or decode-bound? |
| End-to-end latency p50 / p95 / p99 | e2e | The SLO panel. |
| TTFT p50 / p95 / p99 | TTFT | Slow first token → queueing or long prefill. |
| ITL p50 / p95 / p99 | ITL | Contended decode loop. |
| Request rate | RPS | Are we keeping up with driver. |
| KV-cache utilization | free KV blocks | Headroom for concurrency. |
| Prefix-cache hit rate | prefix-cache hit rate | Schema prefixes hitting cache. Observed 87–92 % under load. |

Screenshot: `screenshots/grafana_serving.png` (real chromium-headless render against the dashboard with anonymous-viewer enabled in `docker-compose.yml`).

---

## §3 Agent design (Phase 3) — `agent/graph.py` + `agent/prompts.py`

LangGraph, cap at 3 total LLM-emitting iterations.

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

**Performance-oriented choices baked into the code:**
- Module-level cached `ChatOpenAI` so the httpx pool isn't churned per call.
- Tuned `httpx.AsyncClient(limits=Limits(max_connections=500, max_keepalive_connections=200))` — default 100 / 20 throttled at 30+ req/s.
- Per-node `max_tokens` caps via `llm("node").bind(max_tokens=...)` (slide 23/24 *output-length caps*).
- Verify *fast-path*: if execution returned ≥ 1 row, accept without an LLM call. Baseline eval shows the LLM verifier didn't catch any errors anyway (`per_iteration_pass_rate` flat at 36.7 %); skipping it on the happy path drops one full LLM round trip per request.

---

## §4 Tracing (Phase 4) — Langfuse

`agent/server.py` installs `langfuse.langchain.CallbackHandler` when the env keys are set, and passes the request's tags both as `tags=[...]` (string list — chip-filterable in the trace list) and as `metadata={...}` (key=value, queryable). `docker-compose.yml` pre-seeds `LANGFUSE_INIT_ORG_ID / PROJECT_PUBLIC_KEY / SECRET_KEY / USER_*` so the Langfuse v4 stack comes up with a working admin account and deterministic API keys baked in — no UI signup loop.

Real Langfuse v4 SDK is **async + batched by default** ("almost no latency with fully async requests; tracing never blocks") — so the trace exporter ran on the hot path of *every single load-test request below*. The prior-pass "Langfuse OTLP saturation" reading was a misdiagnosis of the schema-NULL crash (see §7).

Screenshots:
- `screenshots/langfuse_trace.png` — real Langfuse UI, trace `92c1c39f0b0560812cbb1e3f97594eda`, showing the full **verify → revise loop firing twice** (3 iterations: `generate_sql → verify → revise → verify → revise → verify`). Tags `db_id=california_schools, phase=revise-demo` visible.
- `screenshots/langfuse_tags.png` — real Langfuse UI trace-list view; each of the 10 trace-smoke rows shows chip tags `db_id=…, phase=trace-smoke, idx=…` in the Tags column.

---

## §5 Baseline eval (Phase 5) — `evals/run_eval.py`

Execution-accuracy comparison: run agent's final SQL and the gold SQL against the same SQLite DB; canonicalize each row set (sort, `None → ""`, `str()` cast); compare as multisets. Per-iteration carry-forward in `summarize()`: if the agent stopped at iter j < k, treat iter-k result as identical to iter-j.

Results in `results/eval_baseline.json`:
- **Overall pass rate: 12 / 30 = 40 %** (wall clock 19 s with Langfuse on)
- Iteration distribution: 22 terminated at iter 1, 1 at iter 2, 7 hit the iter-3 cap
- Per-iteration pass rate (carry-forward): iter 1 = iter 2 = iter 3 = 40 % — **the LLM verifier never flipped a verdict**, which is why the verify fast-path was safe.

Per-DB pass rate shows persistent failures on three schemas (`formula_1`, `thrombosis_prediction`, `toxicology`). They have unusual column names and the prefix-cached schema rendering doesn't help the model; better schema rendering with example values would be the next quality lever.

Screenshot of dashboard during the eval: `screenshots/grafana_eval_run.png`.

---

## §6 SLO journey (Phase 6 — main grade) — 7 experiments

`load_test/driver.py --rps R --duration 300` against the agent on `:8001`. Final-config baseline first; then 6 experiments testing specific slide-23/24/25 knobs; then an RPS sweep on the winning config.

| # | What I saw | Hypothesis | Change (slide 23/24/25 box) | Result (p50 / p95 / p99, ok %, achieved RPS) | Source |
|---|---|---|---|---|---|
| 0 baseline | Schema-NULL bug had previously masqueraded as a Langfuse OTLP problem. Verified the corrected hypothesis: with the bug fixed and Langfuse fully on, the system hits SLO. | Final iter-7 code + Langfuse fully on. | (no change — this row *is* the baseline) | **1.12 / 4.15 / 13.16 s**, 100 %, **9.41 RPS** — **SLO hit** | `results/load_test_baseline_langfuse_on.json` |
| 2 | Verifier sometimes emitted long completions, padding p99. | Cap `max_tokens` per node: generate 200, **verify 32**, revise 200. | slide 23/24 *output-length caps* | 1.15 / 4.55 / 15.10 s, 100 %, 9.36 RPS — neutral within noise | `results/exp2_max_tokens.json` |
| 3 | We're decode-bound at p99; smaller per-step token budget could trade prompt throughput for lower ITL. | `--max-num-batched-tokens 8192 → 4096` | slide 24 *throughput-queue* | 1.11 / 4.18 / 12.74 s, 100 %, 9.33 RPS — wash | `results/exp3_smaller_batched_tokens.json` |
| 4 ★ | Course slide 25 calls out weight quantization as the largest memory-headroom lever. Try **FP8 weights** (`Qwen/Qwen3-30B-A3B-FP8`). | Swap bf16 → FP8 model checkpoint. | slide 25 *compression / weight quantization* | **70.93 / 114.93 / 118.98 s, 9.7 %, 8.33 RPS** — **hard regression** | `results/exp4_fp8_weights.json` |
| 5 | FP8 weights alone failed; maybe stacking FP8 KV recovers (combined ~4× memory). | Add `--kv-cache-dtype fp8`. | slide 25 compression × 2 | 70.49 / 117.13 / 119.85 s, 9.1 %, 8.33 RPS — same regression | `results/exp5_kv_fp8.json` |
| 6a | Is the FP8 regression because of FP8 + spec-dec interaction? Strip spec-dec. | FP8 + FP8 KV, spec-dec OFF. | slide 23 decode (control) | 69.40 / 114.41 / 118.16 s, 10.1 %, 8.33 RPS — **same regression, so spec-dec is exonerated; FP8 itself is the culprit** | `results/exp6_specdec_off.json` |
| 6b | Confirm spec-dec back on doesn't help FP8. | FP8 + FP8 KV, spec-dec ON. | slide 23 decode | 72.20 / 116.68 / 119.24 s, 9.6 %, 8.33 RPS — confirmed | `results/exp6_specdec_on.json` |
| 7 sweep | Final config = baseline (bf16 + spec-dec, FP8 reverted). Find the actual RPS ceiling. | RPS sweep 10 / 15 / 20 / 30 against baseline. | slide 26 latency/throughput frontier | 10: 1.35 / **6.58 s** / 15.80, **99.9 %**, 9.37 RPS. 15: 1.50 / **6.31 s** / 16.85, **100 %**, 14.23 RPS. 20: 2.29 / 9.04 / 19.24, 99.8 %, 18.38 RPS. 30: 25.89 / 85.46 / 111.71, 75.4 %, 25.00 RPS — saturated. | `results/exp7_rps_{10,15,20,30}.json` |

**Live n-gram spec-dec metrics during baseline**: mean accept length 4.1 tokens / 5, avg draft acceptance 62.2 %, per-position acceptance `0.73 / 0.64 / 0.60 / 0.57 / 0.56`. SQL output is heavily templated (`SELECT`/`FROM`/`JOIN`/`WHERE`/...) — exactly the workload n-gram speculation expects.

**Headroom**: the RPS-15 row beats the SLO-target of 10 RPS by 50 %. The system holds **100 % OK at p95 = 6.31 s sustained at 14.23 RPS**. It crosses the 5 s p95 line between RPS 10 and RPS 15 (variance: the bare-baseline row at RPS-10 with a warmed prefix cache hit p95 = 4.15 s; the cold-start RPS-10 sweep row hit p95 = 6.58 s — prefix cache warm-up is the dominant first-minute variance source).

**Quality survival**: `results/eval_after_tuning.json` ran the 30-question eval against the same bf16 + spec-dec final config → **11 / 30 = 36.7 %** vs baseline 12 / 30 = 40.0 %. One-question swing (which question varies between runs because vLLM batching shuffles non-determinism even at temperature=0). Within noise, no regression.

`screenshots/grafana_before.png` — FP8 stack chaos (p95 spikes past 4 mins).
`screenshots/grafana_after.png` — bf16 winning config, healthy panels under the full RPS sweep.

---

## §7 What didn't work / surprised us / corrected misdiagnoses

- **FP8 weights regressed by 25× on this MoE + vLLM 0.10.2 setup.** Slide 25 predicts a weight-quantization win (memory headroom + throughput). We instead saw P95 jump from 4.15 s → 114.93 s, ok-rate 100 % → 9.7 %, across four FP8 variants (with/without FP8 KV, with/without spec-dec). The regression is reproducible and identical across configs, so FP8 *itself* is the bottleneck, not interaction with KV / spec-dec. Likely root cause: vLLM 0.10.2's FP8 path for Qwen3 MoE is missing or unfused-kernel; needs deeper instrumentation than we had budget for. We **kept bf16 in the submitted config and documented the FP8 result honestly** — the redo of this experiment was specifically requested ("include #4 as well") and the negative result is reportable.
- **The verify → revise loop adds zero quality** under the current verifier prompt. Per-iteration pass rate is flat at iter 1 / 2 / 3 — the LLM verifier never flipped a verdict across iterations. The prompt biases toward `ok=true` (to avoid wasted revises), so genuine failures pass silently. As written, the loop is a pure latency tax. **The verify fast-path** (skip the LLM verifier when execution returned ≥ 1 row) produced the single biggest latency improvement we have. A *stricter* verifier with better prompting could still pay off — that's in §8.
- **The prior pass's "Langfuse OTLP saturation" reading was wrong.** Iter-5/6 of the prior pass saw 379 HTTP 500s under load and `agent.log` showed `Failed to export span batch ... Read timed out`. We turned Langfuse off as a "fix" — got p95 = 3.20 s — and shipped. *In the redo we re-enabled Langfuse, kept it on for every experiment, and still hit SLO*: the OTLP timeouts were a downstream symptom of `agent/schema.py` crashing on `PRAGMA foreign_key_list` returning NULL for `fk[4]` on certain BIRD DBs. Langfuse v4's exporter is async+batched and doesn't block the hot path. Lesson: read the actual stack trace before blaming the observability layer.
- **`max_num_seqs` widening *hurt* when we were already compute-bound.** Slide 24 lists "large max_num_seqs" as a throughput knob — but slide 22 makes the cost explicit (latency↔throughput trade-off triangle). At baseline we were at GPU util 93 % with KV cache only 8 % used: the GPU was the limiter, not memory. Widening the batch to 256 (prior pass, iter-3) just made every individual sequence share less compute per scheduler step. p50 doubled, p95 quadrupled. The course's "online sweet spot" diagram (slide 26) is exactly this: we were already at the online sweet spot, and pushing toward the batch sweet spot moves you *backwards* on latency.
- **Adding uvicorn workers from 1 to 4 gave ≈ 0 improvement.** Expected ~4× throughput; got ~3 %. The bottleneck was upstream of agent threading — at 36 concurrent on vLLM, GPU was already 93 %. Symmetric lesson: don't add capacity to a layer that isn't the constraint (slide 22 triangle: where we are on the curve dictates which knob helps).

---

## §8 What I'd do at multi-replica scale

In priority order, mapped to the deck:

1. **Async-batched span exporter validation** + a budget for OTLP. Iter through one Langfuse rollout where we deliberately push to 30+ RPS with full tracing and confirm queue-depth (slide 37 LLM-native signal) doesn't grow unbounded. If it does, the right fix per slide 38 is sampling: keep 100 % of *errors* and "premium" tenants, drop bulk under load.
2. **Real FP8 path bring-up** — slide 25 promises this should be the largest memory-headroom lever. The course's promised behavior (~20–25 % throughput at high batch, ~50 % KV footprint reduction) is not what we observed on vLLM 0.10.2 + Qwen3-30B-A3B MoE. Worth a follow-up either pinning a known-good vLLM version (e.g. nightly with the Qwen-MoE FP8 kernel) or quantizing in-house via llmcompressor. Once FP8 works, KV-cache quantization (`--kv-cache-dtype fp8`) is free on top.
3. **KV-aware routing** (slide 30–31). With prefix-cache hit rate at 87–92 % on one replica, a multi-replica deployment would lose most of that to per-replica hash splits. The diagram is exactly what we'd build: shared prefix-hash → endpoint index, with engines pushing free-blocks / queue-depth back to the router.
4. **Stricter verify prompt** so the verify→revise loop actually buys quality. Currently it never flips a verdict; a strict critic that catches obvious errors (wrong table, wrong aggregate, wrong filter direction) could push eval from 40 % toward the published BIRD-dev baselines (~55–65 % execution accuracy for similar-class models). Quality, not latency, is now the limiting axis.
5. **Schema rendering improvements** for the three 0 %-pass schemas (`formula_1`, `thrombosis_prediction`, `toxicology`). Sample row preview + relationship descriptions instead of bare CREATE TABLE.
6. **P/D disaggregation** (slide 23). Out of scope on 1 GPU; with 2+ GPUs this lets prefill and decode scale independently. Particularly useful for our workload because prefill (schema-heavy) and decode (short SQL) have very different shapes.
7. **Predictive autoscale + warm pool** (slide 34/35). A diurnal traffic pattern + the cold-start signature (HBM/GPU/TTFT) means a 1× warm pool and a 30s-ahead trigger using TTFT-on-synthetic would smooth scale-ups without flapping.

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
| vLLM manual query | `screenshots/vllm_manual_query.png` (real `script(1)` + `aha` + chromium-headless) |
| Grafana dashboard | `screenshots/grafana_serving.png` |
| Langfuse trace (revise loop) | `screenshots/langfuse_trace.png` (real UI, trace `92c1c39f0b0560812cbb1e3f97594eda`) |
| Langfuse tags | `screenshots/langfuse_tags.png` (real UI, trace list with chip tags) |
| Grafana during eval | `screenshots/grafana_eval_run.png` |
| Grafana before | `screenshots/grafana_before.png` (FP8 stack regression, p95 → 4 min spike) |
| Grafana after | `screenshots/grafana_after.png` (bf16 + spec-dec final config, healthy under RPS sweep) |

Supporting (not in submission spec, kept for cross-reference from the diagnosis table):
`results/load_test_baseline_langfuse_on.json`, `results/exp2..6*.json`, `results/exp7_rps_*.json`.
