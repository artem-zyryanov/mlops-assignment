"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def _score_iterations(
    history: list[dict],
    db_id: str,
    gold_rows: list[tuple] | None,
) -> list[dict]:
    """Replay the agent's per-iteration SQL against the DB and score each one.

    Returns a list ordered by iteration (1-indexed). Each entry records the SQL
    the agent had at that point and whether running it against the DB would
    have matched the gold result. We use this for the "if we had stopped at
    iter k" pass rate in summarize().
    """
    out: list[dict] = []
    for h in history or []:
        sql = h.get("sql")
        if sql is None:
            continue
        ok, rows, err = run_sql(db_id, sql)
        out.append({
            "sql": sql,
            "execution_ok": ok,
            "execution_error": err,
            "passed": ok and matches(gold_rows, rows),
        })
    return out


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    t0 = time.monotonic()
    try:
        resp = httpx.post(
            agent_url,
            json={"question": question["question"], "db": question["db_id"]},
            timeout=120.0,
        )
        resp.raise_for_status()
        body = resp.json()
        agent_error: str | None = None
    except Exception as e:  # noqa: BLE001
        body = {}
        agent_error = f"{type(e).__name__}: {e}"
    latency_seconds = time.monotonic() - t0

    # Gold rows: run gold_sql against the same DB.
    gold_ok, gold_rows, gold_err = run_sql(question["db_id"], question["gold_sql"])

    pred_rows_raw = body.get("rows")
    # The agent returns rows as list[list]; canonicalize() takes list[tuple].
    pred_rows: list[tuple] | None = (
        [tuple(r) for r in pred_rows_raw] if pred_rows_raw is not None else None
    )
    final_passed = (
        agent_error is None
        and body.get("ok", False)
        and gold_ok
        and matches(gold_rows, pred_rows)
    )

    iter_scores = _score_iterations(body.get("history", []), question["db_id"], gold_rows)

    return {
        "question": question["question"],
        "db_id": question["db_id"],
        "gold_sql": question["gold_sql"],
        "gold_ok": gold_ok,
        "gold_error": gold_err,
        "predicted_sql": body.get("sql", ""),
        "agent_ok": body.get("ok", False),
        "agent_error": agent_error or body.get("error"),
        "iterations": body.get("iterations", 0),
        "latency_seconds": latency_seconds,
        "passed": final_passed,
        "per_iteration": iter_scores,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    total = len(results)
    overall_passed = sum(1 for r in results if r["passed"])

    # Per-iteration pass rate, carry-forwarded. We compute up to the max
    # iteration count actually observed across results.
    max_iters = max((len(r.get("per_iteration", [])) for r in results), default=0)
    per_iteration_pass = []
    for k in range(1, max_iters + 1):
        passed_at_k = 0
        for r in results:
            iters = r.get("per_iteration", [])
            if not iters:
                continue
            # Carry-forward: if agent stopped before iter k, reuse its final.
            idx = min(k, len(iters)) - 1
            if iters[idx].get("passed"):
                passed_at_k += 1
        per_iteration_pass.append({
            "iteration": k,
            "passed": passed_at_k,
            "pass_rate": passed_at_k / total if total else 0.0,
        })

    # Latency percentiles over successful agent calls.
    latencies = sorted(
        r["latency_seconds"] for r in results if r.get("latency_seconds") is not None
    )

    def pct(p: float) -> float:
        if not latencies:
            return float("nan")
        k = int(round(p * (len(latencies) - 1)))
        return latencies[k]

    # Per-DB pass rate so we can spot pathological schemas in REPORT.
    by_db: dict[str, dict] = {}
    for r in results:
        d = by_db.setdefault(r["db_id"], {"total": 0, "passed": 0})
        d["total"] += 1
        if r["passed"]:
            d["passed"] += 1
    per_db = [
        {"db_id": k, "total": v["total"], "passed": v["passed"],
         "pass_rate": v["passed"] / v["total"] if v["total"] else 0.0}
        for k, v in sorted(by_db.items())
    ]

    return {
        "total": total,
        "passed": overall_passed,
        "pass_rate": overall_passed / total if total else 0.0,
        "iteration_distribution": _iteration_distribution(results),
        "per_iteration_pass_rate": per_iteration_pass,
        "latency_p50_seconds": pct(0.50),
        "latency_p95_seconds": pct(0.95),
        "latency_max_seconds": latencies[-1] if latencies else float("nan"),
        "per_db_pass_rate": per_db,
    }


def _iteration_distribution(results: list[dict]) -> dict[str, int]:
    """How many questions terminated after 1, 2, 3, ... iterations."""
    dist: dict[str, int] = {}
    for r in results:
        key = str(r.get("iterations", 0))
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items()))


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    parser.add_argument("--limit", type=int, default=None,
                        help="optional: only run the first N questions (smoke test)")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
