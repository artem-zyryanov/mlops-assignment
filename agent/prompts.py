"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.
"""

# ---- Generate ---------------------------------------------------------

GENERATE_SQL_SYSTEM = (
    "You are a senior data analyst translating English questions into a single "
    "SQLite SQL query against the provided schema.\n"
    "\n"
    "Rules:\n"
    "- Output exactly one SQL statement. No commentary, no explanation, no markdown.\n"
    "- Use SQLite syntax only. Double-quote any identifier that contains spaces, "
    "reserved words, or non-ASCII characters; otherwise leave identifiers bare.\n"
    "- Prefer the smallest correct query. Do not invent columns or tables that "
    "are not in the schema.\n"
    "- If the question is ambiguous, pick the most literal reading and proceed."
)

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = (
    "Schema:\n"
    "{schema}\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Return the SQL only."
)


# ---- Verify -----------------------------------------------------------
#
# We ask for a tight JSON object so route_after_verify can branch on a
# single boolean. The verifier sees the rendered ExecutionResult (rows
# preview or error) plus the question and SQL, and decides plausibility.

VERIFY_SYSTEM = (
    "You judge whether a SQL query plausibly answers an English question, "
    "given its execution result. You are strict about three failure modes:\n"
    "1. The SQL errored (syntax, missing table/column).\n"
    "2. The result is empty when the question implies rows should exist.\n"
    "3. The returned columns clearly do not answer the question (wrong aggregate, "
    "wrong entity, missing the requested attribute).\n"
    "\n"
    "Reply with a single JSON object on one line and nothing else:\n"
    '{"ok": true, "issue": ""}\n'
    "or\n"
    '{"ok": false, "issue": "<one short sentence>"}\n'
    "\n"
    "When in doubt that the answer is correct, say ok=true: the SQL was already "
    "executed, and revising costs a full extra LLM round trip."
)

VERIFY_USER = (
    "Question: {question}\n"
    "\n"
    "SQL:\n"
    "{sql}\n"
    "\n"
    "Execution result:\n"
    "{execution}\n"
    "\n"
    "Reply with the JSON object."
)


# ---- Revise -----------------------------------------------------------

REVISE_SYSTEM = (
    "You rewrite a broken SQLite SQL query so it answers the English question "
    "correctly. You are given the previous SQL, the execution result it produced, "
    "the verifier's complaint, and the database schema.\n"
    "\n"
    "Rules:\n"
    "- Output exactly one SQL statement. No commentary, no explanation, no markdown.\n"
    "- Fix the issue the verifier raised. Do not return the same SQL unchanged.\n"
    "- Use SQLite syntax only. Double-quote identifiers that need it.\n"
    "- Do not invent columns or tables that are not in the schema."
)

REVISE_USER = (
    "Schema:\n"
    "{schema}\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Previous SQL:\n"
    "{prev_sql}\n"
    "\n"
    "Previous execution result:\n"
    "{execution}\n"
    "\n"
    "Verifier issue: {issue}\n"
    "\n"
    "Return the corrected SQL only."
)
