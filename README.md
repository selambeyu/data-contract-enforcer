# Data Contract Enforcer

Schema integrity and lineage attribution system for a 5-week multi-system data platform.
Retroactively writes and enforces Bitol data contracts across Week 1–5 outputs.

## Architecture

```
Week 3 extractions ──┐
Week 4 lineage     ──┼──► ContractGenerator ──► generated_contracts/*.yaml
Week 5 events      ──┤         │                         │
LangSmith traces   ──┘         │                  schema_snapshots/{id}/{ts}.yaml
                               │                  generated_contracts/*_dbt.yml
                               ▼
                    ContractRegistry                 SchemaEvolutionAnalyzer
                (subscriptions.yaml)                 (producer-side CI gate)
                         │                                    │
                         └──────────► ValidationRunner ◄──────┘
                                      (AUDIT/WARN/ENFORCE)
                                              │
                                              ▼
                                    ViolationAttributor
                             (BFS lineage + git blame + blast radius)
                                              │
                                    violation_log/violations.jsonl
                                              │
                                    AIExtensions + ReportGenerator
                                              │
                                    enforcer_report/report_data.json
```

**Trust tiers**:
- Tier 1 = our systems (full git + lineage traversal)
- Tier 3 = external (LangSmith — contract-version attribution only, no git)

**Enforcement modes** (`--mode`):
- `AUDIT`   — log only, never block [default — use on first deployment]
- `WARN`    — block on CRITICAL only
- `ENFORCE` — block on CRITICAL or HIGH

## Quick Start (fresh clone)

```bash
# 0. Install dependencies
uv sync

# 1. Run migration scripts (transform source data to canonical JSONL)
uv run outputs/migrate/week3_ledger_to_extractions.py
uv run outputs/migrate/week4_graph_to_snapshot.py
uv run outputs/migrate/week5_events_to_jsonl.py
uv run outputs/migrate/traces_export_to_jsonl.py

# 2. Generate contracts (writes YAML + dbt schema.yml + timestamped snapshot)
uv run src/contracts/generator.py \
  --source outputs/week3/extractions.jsonl \
  --contract-id week3-document-refinery-extractions \
  --lineage outputs/week4/lineage_snapshots.jsonl \
  --output generated_contracts/

uv run src/contracts/generator.py \
  --source outputs/week4/lineage_snapshots.jsonl \
  --contract-id week4-brownfield-cartographer-lineage \
  --lineage outputs/week4/lineage_snapshots.jsonl \
  --output generated_contracts/

uv run src/contracts/generator.py \
  --source outputs/week5/events.jsonl \
  --contract-id week5-event-sourcing-platform-events \
  --lineage outputs/week4/lineage_snapshots.jsonl \
  --output generated_contracts/

uv run src/contracts/generator.py \
  --source outputs/traces/runs.jsonl \
  --contract-id langsmith-trace-runs \
  --lineage outputs/week4/lineage_snapshots.jsonl \
  --output generated_contracts/

# 3. Validate baseline data (AUDIT mode — never blocks)
uv run src/contracts/runner.py \
  --contract generated_contracts/week3-document-refinery-extractions.yaml \
  --data outputs/week3/extractions.jsonl \
  --baselines schema_snapshots/baselines.json \
  --output validation_reports/week3_baseline.json \
  --mode AUDIT

# Run in ENFORCE mode to gate a deployment:
uv run src/contracts/runner.py \
  --contract generated_contracts/week3-document-refinery-extractions.yaml \
  --data outputs/week3/extractions.jsonl \
  --baselines schema_snapshots/baselines.json \
  --output validation_reports/week3_enforce.json \
  --mode ENFORCE

# 4. Inject test violations and validate
uv run outputs/migrate/inject_violations.py

uv run src/contracts/runner.py \
  --contract generated_contracts/week3-document-refinery-extractions.yaml \
  --data outputs/week3/extractions_violated.jsonl \
  --baselines schema_snapshots/baselines.json \
  --output validation_reports/violated_week3.json \
  --mode WARN

uv run src/contracts/runner.py \
  --contract generated_contracts/week5-event-sourcing-platform-events.yaml \
  --data outputs/week5/events_violated.jsonl \
  --baselines schema_snapshots/baselines.json \
  --output validation_reports/violated_week5.json \
  --mode AUDIT

# 5. Attribute violations — registry blast radius + BFS lineage + git blame
uv run src/contracts/attributor.py \
  --violation validation_reports/violated_week3.json \
  --registry contract_registry/subscriptions.yaml \
  --lineage outputs/week4/lineage_snapshots.jsonl \
  --output violation_log/violations.jsonl \
  --codebase-root .

# 6. Schema evolution analysis (producer-side CI gate)
#    Spec CLI: --contract-id + --since
uv run src/contracts/schema_analyzer.py \
  --contract-id week3-document-refinery-extractions \
  --since "7 days ago" \
  --snapshots schema_snapshots \
  --output validation_reports/schema_evolution_week3-document-refinery-extractions.json

# Or diff all contracts at once:
uv run src/contracts/schema_analyzer.py \
  --all \
  --snapshots schema_snapshots \
  --output-dir validation_reports

# 7. AI extensions (embedding drift + prompt schema + LLM output violation rate)
uv run src/contracts/ai_extensions.py \
  --mode all \
  --extractions outputs/week3/extractions.jsonl \
  --traces outputs/traces/runs.jsonl \
  --verdicts outputs/week2/verdicts.jsonl \
  --output enforcer_report/ai_extensions.json

# 8. Generate final report
uv run src/contracts/report_generator.py \
  --validation-dir validation_reports \
  --violations violation_log/violations.jsonl \
  --diffs-dir validation_reports \
  --ai-extensions enforcer_report/ai_extensions.json \
  --registry contract_registry/subscriptions.yaml \
  --output enforcer_report/report_data.json
```

## Key Findings

| Contract | Checks | Status | Open Issues |
|---|---|---|---|
| week3-document-refinery-extractions | 29 | ✅ PASS (baseline) | None on clean data |
| week4-brownfield-cartographer-lineage | 20 | ⚠️ 1 CRITICAL | node_id uses path strings not UUID (open design violation) |
| week5-event-sourcing-platform-events | 27 | ✅ PASS (baseline) | — |
| langsmith-trace-runs | 34 | ✅ PASS (baseline) | Synthesised data (no real LangSmith export) |

**Violation detection** (injected tests pass):
- Week 3 violated: `fact_confidence × 100` → `HIGH` range + statistical drift (caught by ENFORCE mode)
- Week 5 violated: `event_type = "InvalidEventXYZ"` → `CRITICAL` enum_conformance

## Project Structure

```
src/contracts/
  generator.py        # Stages 1–4: profile → Bitol YAML + dbt schema.yml + timestamped snapshot
  runner.py           # 7 checks, AUDIT/WARN/ENFORCE modes, dot-scoped check_id
  attributor.py       # BFS lineage traversal + confidence_score blame chain + registry blast radius
  schema_analyzer.py  # --contract-id/--since CLI, diff snapshots, BREAKING/COMPATIBLE taxonomy, rollback plan
  ai_extensions.py    # OpenAI embedding drift + prompt input schema + LLM output violation rate
  report_generator.py # Spec health score: (passed/total)×100 −20/CRITICAL −10/HIGH

contract_registry/
  subscriptions.yaml  # 6 subscriptions with breaking_fields as {field, reason} objects + validation_mode

generated_contracts/  # *.yaml Bitol contracts + *_dbt.yml dbt schema counterparts
schema_snapshots/     # {contract_id}/{timestamp}.yaml — written on every generator run
validation_reports/   # Per-run JSON reports + schema_evolution_*.json diffs
violation_log/        # violations.jsonl — spec-compliant schema with check_id, blame_chain, blast_radius
enforcer_report/      # ai_extensions.json, report_data.json
outputs/
  migrate/            # 5 migration scripts + inject_violations.py
  week3/, week4/, week5/, traces/
```
