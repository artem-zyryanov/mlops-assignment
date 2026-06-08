# MLOps assignment report

Text-to-SQL over BIRD with vLLM serving Qwen3-30B-A3B on one H100, end-to-end SLO target P95 < 5s at 10+ RPS over 5 minutes.

## 1. Serving configuration (Phase 1)

`scripts/start_vllm.sh` runs vLLM with the following flags. Each is justified for the workload (1.5–3K-token schema-heavy prompts, short structured SQL outputs, ~2–3 dependent agent calls per user request).

| Flag | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `1` | Single H100. TP > 1 needs > 1 GPU; nothing to split. |
| `--max-model-len` | `8192` | Schema (1–3K) + question + few-shot fit comfortably. Smaller `max_model_len` ⇒ larger KV-cache budget per slot ⇒ more concurrent requests on the same GPU. |
| `--gpu-memory-utilization` | `0.90` | Leave 10 % for CUDA workspace and cuBLAS scratch; explicit so we know what changed if we revisit. |
| `--enable-prefix-caching` | on | Schema text is byte-identical across requests for the same `db_id`. The 1500-question perf pool hits ~138 BIRD DBs, so prefix caching turns repeated schema prefills into cache lookups — a TTFT win that should grow with load. |
| `--enable-chunked-prefill` | on | Lets prefill and decode share the GPU under continuous batching. Smooths P95 spikes caused by long-schema requests entering the queue. |
| `--max-num-seqs` | `64` *(initial)* | Upper bound on concurrent sequences. Will be tuned in Phase 6. |
| `--max-num-batched-tokens` | `8192` *(initial)* | Per-step token budget. Sized to a small multiple of one full prompt so prefill and decode coexist; tuned in Phase 6. |

Qwen3-30B-A3B is an MoE (30 B total, 3 B active per token). Weights fit on one H100 80GB at FP16 (~60 GB) with ~20 GB headroom for KV cache. We did not quantize for the baseline — see §6 for what we'd try next.

Screenshot: `screenshots/vllm_manual_query.png` (vLLM responds; SQL is sensible).

## 2. Observability (Phase 2)

Dashboard `infra/grafana/provisioning/dashboards/serving.json` covers three categories. Each panel is built to answer one question at 3 AM.

| Panel | Question it answers |
|---|---|
| Requests in flight (running/waiting/swapped) | Is the scheduler saturated? |
| Token throughput (prompt vs gen tok/s) | Are we prefill- or decode-bound? |
| End-to-end request latency (p50/p95/p99) | The SLO panel. |
| TTFT (p50/p95/p99) | Is the first token slow? → prefill or queue. |
| Inter-token latency (p50/p95/p99) | Is decode slow per token? |
| Request rate (success/s) | Are we keeping up with the driver's target RPS? |
| KV-cache utilization | Do we have headroom for more concurrency? |
| Prefix-cache hit rate | Are schema prefixes hitting cache? |

Screenshot: `screenshots/grafana_serving.png` (panels reacting under a burst).

## 3. Agent design (Phase 3)

Graph in `agent/graph.py`. Capped at 3 total LLM-emitting iterations.

```
question + db_id
        │
        ▼
  attach_schema                        (rendered via agent/schema.py; lru_cached)
        │
        ▼
  generate_sql ──── vLLM call #1
        │
        ▼
    execute        (provided; read-only sqlite, 5s timeout)
        │
        ▼
    verify ─────── vLLM call #2        (short-circuits on SQL error → revise)
        │
   ok=true ─► END
        │
   ok=false ──► revise ─── vLLM call #3 ─► execute ─► verify (loop)
                                                            │
                                  iteration == MAX_ITERATIONS ─► END
```

**Prompts** (`agent/prompts.py`):
- `GENERATE_SQL` instructs SQLite syntax only, single statement, no markdown, double-quote unusual identifiers.
- `VERIFY` asks for a one-line JSON `{"ok": bool, "issue": str}`. The verifier is told to bias toward `ok=true` when uncertain, because each `ok=false` costs a full extra vLLM round trip. `verify_node` short-circuits on execution errors so we don't burn a model call asking "is `OperationalError` a good answer?".
- `REVISE` includes schema + question + prior SQL + rendered execution result + verifier issue, and is told to never return the same SQL unchanged.

All three prompts run at `temperature=0` so eval is reproducible.

## 4. Tracing (Phase 4)

Langfuse v4 callback wired in `agent/server.py`; tags from the request body propagate as trace metadata. The Phase 6 iteration log relies on filtering Langfuse traces by `db_id` and iteration count.

Screenshots: `screenshots/langfuse_trace.png` (single trace waterfall: `generate_sql → execute → verify → revise → execute → verify`), `screenshots/langfuse_tags.png` (trace list with tags visible).

## 5. Baseline eval (Phase 5)

Eval harness `evals/run_eval.py`. Execution-accuracy comparison: run both the agent's final SQL and the gold SQL against the same SQLite DB, canonicalize each row set (sort rows, `None → ""`, coerce cells to str), compare as ordered lists. Per-iteration carry-forward in `summarize()`: if the agent stops at iter j < k, treat its iter-k result as identical to iter-j.

Results in `results/eval_baseline.json`.

- Overall pass rate: **TBD**
- Iteration distribution (how many questions terminated at iter 1, 2, 3): **TBD**
- Per-iteration pass rate (after iter k, with carry-forward):

| Iteration | Pass rate |
|---|---|
| 1 | TBD |
| 2 | TBD |
| 3 | TBD |

If iter 1 ≈ iter 3, the loop is not earning its keep — see §6 for what we'd cut.

Screenshot: `screenshots/grafana_eval_run.png`.

## 6. SLO journey (Phase 6 — main grade)

Baseline load test: `python load_test/driver.py --rps 10 --duration 300`.

| # | What I saw | Hypothesis | Change | Result |
|---|---|---|---|---|
| 0 (baseline) | TBD | — | — | TBD |
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |

Screenshots: `screenshots/grafana_before.png`, `screenshots/grafana_after.png`.

**Final**: P95 = **TBD** s, achieved RPS = **TBD** over **TBD** s. SLO: **TBD** (hit/miss with gap).

**Quality survival**: re-running eval on the tuned config → `results/eval_after_tuning.json`. Δ pass rate vs baseline = **TBD**.

## 7. What didn't work

*Be honest here.* TBD — at least one experiment that backfired (e.g. raising `max_num_seqs` past the KV-cache headroom and tipping into evictions).

## 8. What I'd do with more time

Specific, in priority order:

1. **AWQ or FP8 quantization** of Qwen3-30B-A3B. FP8 on H100 is roughly free on quality for this size class, halves weight memory, gives back ~30 GB to the KV cache, and lets us double `max_num_seqs`. Highest expected SLO leverage of any single change.
2. **Bypass the verifier on easy cases**. If the generated SQL parses and returns ≥ 1 row without error, skip the LLM verify step entirely. The verifier is the most expensive node on the critical path; cutting it on the easy-majority cases is a one-LLM-call reduction in median latency.
3. **Speculative decoding** with a small draft model (e.g. a Qwen3-0.6B served on the same GPU). SQL is highly templated — high acceptance rate likely; ITL falls.
4. **Per-`db_id` rendered-schema cache** in the agent process. `agent/schema.py` already `lru_cache`s, but the agent restarts wipe it; persisting to disk and warming on boot makes the prefix-cache hit rate near-100 % from request 1.
5. **Two replicas behind a small load balancer**. Even sharing one H100 via MIG slicing (if supported on this SKU) doubles slot count for short queries; cheaper than a second GPU.

---

*All numbers in §5 and §6 are filled in after the H100 booking window.*
