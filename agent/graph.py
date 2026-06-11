"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# 3-5 is a reasonable range; tune it as part of Phase 3.
MAX_ITERATIONS = 3

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


_LLM: ChatOpenAI | None = None


def llm() -> ChatOpenAI:
    """Cached chat client. One per worker process.

    Construction was previously per-call, which spun up a fresh httpx
    AsyncClient (and its connection pool) on every LLM round-trip; under load
    this churns connections and bottlenecks before vLLM. Module-level cache
    keeps the connection pool warm.
    """
    global _LLM
    if _LLM is None:
        _LLM = ChatOpenAI(
            model=VLLM_MODEL,
            base_url=VLLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0.0,
            max_tokens=256,
        )
    return _LLM


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


async def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Async because graph.ainvoke() is used by the server; sync invoke under load
    serializes each request on the FastAPI threadpool.
    """
    response = await llm().ainvoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_verify_reply(text: str) -> tuple[bool, str]:
    """Pull {"ok": bool, "issue": str} out of an LLM reply, defensively.

    The model may wrap the JSON in prose or markdown fences. We try strict JSON
    first, then the first {...} block, and finally fall back to ok=True so a
    malformed verifier reply doesn't trap the loop in pointless revises.
    """
    candidates: list[str] = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    match = _JSON_OBJ_RE.search(text)
    if match:
        candidates.append(match.group(0))

    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "ok" in obj:
            return bool(obj["ok"]), str(obj.get("issue", ""))
    return True, ""


async def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Three-way short-circuit (in order of cheapness):
      1. No execution → fail.
      2. Execution errored → fail with the SQL error (no LLM call).
      3. Execution succeeded with >=1 row → accept (no LLM call).
         Baseline eval showed per-iteration pass rate is flat at 36.7 %,
         i.e. the LLM verifier never flipped a correct result to incorrect
         or vice versa. Skipping it on the happy path cuts ~1 LLM round trip
         per agent call, which is the single biggest SLO win once vLLM is
         GPU-saturated.

    Falls back to the LLM verifier only when execution returned 0 rows,
    where the model might still catch "wrong query, empty result".
    """
    execution = state.execution
    if execution is None:
        return {"verify_ok": False, "verify_issue": "no execution result"}
    if not execution.ok:
        return {
            "verify_ok": False,
            "verify_issue": f"SQL failed: {execution.error}",
            "history": state.history + [
                {"node": "verify", "ok": False, "issue": execution.error}
            ],
        }
    if execution.row_count > 0:
        return {
            "verify_ok": True,
            "verify_issue": "",
            "history": state.history + [
                {"node": "verify", "ok": True, "issue": "", "fast_path": True}
            ],
        }

    response = await llm().ainvoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            question=state.question,
            sql=state.sql,
            execution=execution.render(max_rows=10),
        )),
    ])
    ok, issue = _parse_verify_reply(response.content)
    return {
        "verify_ok": ok,
        "verify_issue": issue,
        "history": state.history + [{"node": "verify", "ok": ok, "issue": issue}],
    }


async def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given verifier feedback."""
    execution_render = state.execution.render(max_rows=10) if state.execution else "(none)"
    response = await llm().ainvoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            prev_sql=state.sql,
            execution=execution_render,
            issue=state.verify_issue,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "revise", "sql": sql}],
    }


def route_after_verify(state: AgentState) -> str:
    """End when verifier is happy or we've burned our iteration budget."""
    if state.verify_ok:
        return "end"
    if state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
