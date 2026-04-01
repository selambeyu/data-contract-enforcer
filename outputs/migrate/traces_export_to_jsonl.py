"""
Migration: src/outputs/traces/runs.json  (single Week 2 agent run dump)
       --> outputs/traces/runs.jsonl     (canonical LangSmith run records)

The source file is a Digital Courtroom agent execution dump.
It has no top-level LangSmith run fields (id, run_type, tokens, cost).
We synthesise canonical records from the nested sub-components:

  Root run       : 1  (the full agent execution)
  Chain runs     : 10 (one per rubric_dimension evaluated)
  Tool runs      : 23 (evidence items: doc_analyst=2, repo_investigator=8,
                                        vision_inspector=1, evidence_aggregator=12)
  LLM runs       : 30 (one per judge opinion)
  ─────────────────
  Total          : 64 records  (>= 50 required)

DEVIATIONS DOCUMENTED:
  - Source has no start_time/end_time → derived from base timestamp + offset
  - Source has no token counts → estimated from content length
  - Source has no total_cost → estimated from token estimates + model rates
  - Source has no run ids → deterministic UUID5 from content hash
  - run_type synthesised from component role (chain/tool/llm)
"""

import json
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path

SRC = Path(__file__).parent.parent.parent / "src" / "outputs" / "traces" / "runs.json"
DST = Path(__file__).parent.parent / "traces" / "runs.jsonl"

_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# Base timestamp for the session (approximate — Week 2 work period)
BASE_TIME = datetime(2026, 3, 20, 9, 0, 0, tzinfo=timezone.utc)

# Approximate cost per 1k tokens (claude-3-sonnet tier)
COST_PER_1K_INPUT  = 0.003
COST_PER_1K_OUTPUT = 0.015


def run_uuid(path: str) -> str:
    return str(uuid.uuid5(_NS, path))


def estimate_tokens(text: str) -> tuple[int, int]:
    """Rough estimate: ~4 chars per token for input, 20% output ratio."""
    char_count = len(str(text))
    input_tokens  = max(10, char_count // 4)
    output_tokens = max(5,  input_tokens // 5)
    return input_tokens, output_tokens


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        (prompt_tokens  / 1000) * COST_PER_1K_INPUT +
        (completion_tokens / 1000) * COST_PER_1K_OUTPUT,
        6
    )


def make_run(
    run_id: str,
    name: str,
    run_type: str,
    inputs: dict,
    outputs: dict,
    start_time: datetime,
    duration_seconds: float,
    parent_run_id: str | None,
    session_id: str,
    tags: list[str],
    error: str | None = None,
) -> dict:
    end_time = start_time + timedelta(seconds=duration_seconds)
    prompt_tokens, completion_tokens = estimate_tokens(
        json.dumps(inputs, default=str)
    )
    total_tokens = prompt_tokens + completion_tokens
    total_cost   = estimate_cost(prompt_tokens, completion_tokens)

    return {
        "id":                str(run_id),
        "name":              name,
        "run_type":          run_type,
        "inputs":            inputs,
        "outputs":           outputs,
        "error":             error,
        "start_time":        start_time.isoformat(),
        "end_time":          end_time.isoformat(),
        "total_tokens":      total_tokens,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_cost":        total_cost,
        "tags":              tags,
        "parent_run_id":     str(parent_run_id) if parent_run_id else None,
        "session_id":        session_id,
        "_migration_meta": {
            "migrated_from":   str(SRC),
            "migration_script": __file__,
            "synthetic":       True,
            "deviations_fixed": [
                "id: synthesised as UUID5 from content path",
                "run_type: inferred from component role (chain/tool/llm)",
                "start_time/end_time: derived from base timestamp + cumulative offset",
                "total/prompt/completion_tokens: estimated from content length (~4 chars/token)",
                "total_cost: estimated from token counts at claude-3-sonnet rates",
            ],
        },
    }


def migrate(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    with src.open() as f:
        data = json.load(f)

    inputs  = data.get("inputs",  {})
    outputs = data.get("outputs", {})
    ls      = data.get("langsmith", {})

    project_id = ls.get("tracing_project", {}).get("id", "unknown")
    session_id = run_uuid(f"session::{project_id}")

    runs: list[dict] = []
    cursor = BASE_TIME  # rolling timestamp

    # ── 1. ROOT RUN (chain) ──────────────────────────────────────────────────
    root_id = run_uuid("root::automaton-auditor::digital-courtroom")
    root_run = make_run(
        run_id        = root_id,
        name          = "DigitalCourtroom.evaluate",
        run_type      = "chain",
        inputs        = {
            "repo_url":  inputs.get("repo_url"),
            "pdf_path":  inputs.get("pdf_path"),
            "rubric_dimensions_count": len(inputs.get("rubric_dimensions", [])),
        },
        outputs       = {
            "report_path":    outputs.get("report_path"),
            "criteria_count": len(outputs.get("final_report", {}).get("criteria", [])),
        },
        start_time    = cursor,
        duration_seconds = 420.0,
        parent_run_id = None,
        session_id    = session_id,
        tags          = ["week2", "digital-courtroom", "root"],
    )
    runs.append(root_run)
    cursor += timedelta(seconds=1)

    # ── 2. CHAIN RUNS — one per rubric dimension ─────────────────────────────
    rubric_dims = inputs.get("rubric_dimensions", [])
    criteria    = outputs.get("final_report", {}).get("criteria", [])
    criteria_map = {c["dimension_id"]: c for c in criteria}

    for i, dim in enumerate(rubric_dims):
        dim_id  = dim.get("id", f"dim_{i}")
        dim_run_id = run_uuid(f"chain::dimension::{dim_id}")
        criterion  = criteria_map.get(dim_id, {})
        dim_run = make_run(
            run_id        = dim_run_id,
            name          = f"EvaluateDimension.{dim.get('name', dim_id)}",
            run_type      = "chain",
            inputs        = {
                "dimension_id":        dim_id,
                "target_artifact":     dim.get("target_artifact"),
                "forensic_instruction": dim.get("forensic_instruction", "")[:200],
            },
            outputs       = {
                "final_score":  criterion.get("final_score"),
                "remediation":  criterion.get("remediation", "")[:200],
            },
            start_time    = cursor,
            duration_seconds = 35.0 + i * 2,
            parent_run_id = root_id,
            session_id    = session_id,
            tags          = ["week2", "dimension-eval", dim.get("target_artifact", "")],
        )
        runs.append(dim_run)
        cursor += timedelta(seconds=36 + i * 2)

    # ── 3. TOOL RUNS — evidence gathering per agent ──────────────────────────
    evidences_by_agent = outputs.get("evidences", {})
    agent_tags = {
        "doc_analyst":          ["week2", "tool", "document-analysis"],
        "repo_investigator":    ["week2", "tool", "git-forensics"],
        "vision_inspector":     ["week2", "tool", "vision-ocr"],
        "evidence_aggregator":  ["week2", "tool", "aggregation"],
    }

    for agent_name, evidence_list in evidences_by_agent.items():
        if not isinstance(evidence_list, list):
            continue
        tags = agent_tags.get(agent_name, ["week2", "tool"])
        for j, ev in enumerate(evidence_list):
            ev_run_id = run_uuid(f"tool::{agent_name}::{j}::{ev.get('goal','')[:40]}")
            ev_run = make_run(
                run_id        = ev_run_id,
                name          = f"{agent_name}.gather[{j}]",
                run_type      = "tool",
                inputs        = {"goal": ev.get("goal"), "location": ev.get("location")},
                outputs       = {
                    "found":      ev.get("found"),
                    "content":    str(ev.get("content", ""))[:300],
                    "confidence": ev.get("confidence"),
                    "rationale":  str(ev.get("rationale", ""))[:200],
                },
                start_time    = cursor,
                duration_seconds = 8.0 + j * 0.5,
                parent_run_id = root_id,
                session_id    = session_id,
                tags          = tags,
                error         = None if ev.get("found") else "evidence_not_found",
            )
            runs.append(ev_run)
            cursor += timedelta(seconds=9 + j * 0.5)

    # ── 4. LLM RUNS — one per judge opinion ──────────────────────────────────
    opinions = outputs.get("opinions", [])
    for k, opinion in enumerate(opinions):
        judge     = opinion.get("judge", f"judge_{k}")
        criterion = opinion.get("criterion_id", "unknown")
        op_run_id = run_uuid(f"llm::opinion::{judge}::{criterion}::{k}")
        op_run = make_run(
            run_id        = op_run_id,
            name          = f"JudgeOpinion.{judge}.{criterion}",
            run_type      = "llm",
            inputs        = {
                "judge":        judge,
                "criterion_id": criterion,
                "cited_evidence_count": len(opinion.get("cited_evidence", [])),
            },
            outputs       = {
                "score":    opinion.get("score"),
                "argument": str(opinion.get("argument", ""))[:300],
            },
            start_time    = cursor,
            duration_seconds = 4.5 + k * 0.2,
            parent_run_id = root_id,
            session_id    = session_id,
            tags          = ["week2", "llm", "judge-opinion", criterion],
        )
        runs.append(op_run)
        cursor += timedelta(seconds=5 + k * 0.2)

    # ── Write ─────────────────────────────────────────────────────────────────
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f_out:
        for run in runs:
            f_out.write(json.dumps(run) + "\n")

    print(f"Migrated {len(runs)} run records: {src} -> {dst}")
    print(f"  root runs    : 1")
    print(f"  chain runs   : {len(rubric_dims)}")
    evidence_count = sum(
        len(v) for v in evidences_by_agent.values() if isinstance(v, list)
    )
    print(f"  tool runs    : {evidence_count}")
    print(f"  llm runs     : {len(opinions)}")


if __name__ == "__main__":
    migrate(SRC, DST)

    # Smoke checks
    with DST.open() as f:
        records = [json.loads(l) for l in f if l.strip()]

    assert len(records) >= 50, f"Expected >=50 records, got {len(records)}"

    required_fields = {"id", "run_type", "start_time", "end_time",
                       "total_tokens", "prompt_tokens", "completion_tokens",
                       "total_cost", "inputs", "outputs", "error",
                       "parent_run_id", "session_id"}

    for r in records:
        missing = required_fields - set(r.keys())
        assert not missing, f"Missing fields in run {r['id']}: {missing}"

    run_types = {r["run_type"] for r in records}
    assert run_types <= {"llm", "chain", "tool", "retriever", "embedding"}, \
        f"Invalid run_type values: {run_types}"

    # end_time > start_time
    from datetime import datetime
    for r in records:
        st = datetime.fromisoformat(r["start_time"])
        et = datetime.fromisoformat(r["end_time"])
        assert et > st, f"end_time <= start_time for run {r['id']}"

    # total_tokens = prompt + completion
    for r in records:
        assert r["total_tokens"] == r["prompt_tokens"] + r["completion_tokens"], \
            f"token mismatch for run {r['id']}"

    print(f"Smoke checks passed. {len(records)} records written.")
    print(f"Run types: { {rt: sum(1 for r in records if r['run_type']==rt) for rt in run_types} }")
