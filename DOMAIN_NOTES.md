# Domain Notes — Data Contract Enforcer

> See [data_flow_diagram.md](data_flow_diagram.md) for the full Mermaid data flow diagram.

---

## Phase 0 Questions

### Q1: Backward-Compatible vs. Breaking Schema Changes

**Three BREAKING changes from my systems:**

1. **Week 3 `doc_id` type change** — `doc_id` in `extraction_ledger.jsonl` is a document name string (e.g. `"Annual_Report_JUNE-2023"`). The canonical contract requires UUID v4. Any consumer that validates format or uses `doc_id` as a UUID foreign key will break silently — it will still run but produce incorrect joins.

2. **Week 4 `edges[].type` → `edges[].relationship` rename** — The Week 4 `module_graph.json` names the edge relationship field `type`. The canonical contract names it `relationship`. Any consumer using `edge['relationship']` receives `KeyError` or `None`. This is breaking because the field was renamed, not added.

3. **Week 4 `node_id` format change** — The Brownfield Cartographer assigns node IDs as composite path strings (e.g. `transformation:/path/to/file.py:181`). The canonical contract's `format: uuid` clause requires `^[0-9a-f]{8}-...-[0-9a-f]{12}$`. This was confirmed as a real FAIL by `ValidationRunner` at run `2026-04-01T08:09:44Z`: all 2,307 node IDs fail the UUID pattern check. Any consumer that treats `node_id` as a UUID foreign key cannot join or reference nodes correctly.

**Three COMPATIBLE changes:**

1. **Adding `cost_actual` to `extraction_ledger.jsonl`** — Some records have `cost_actual` while others do not. This is an optional field addition. Consumers that don't reference it are unaffected.

2. **Adding `block_count` to extraction records** — Present in ledger, absent from canonical schema. Additive non-required field; consumers that ignore unknown fields are unaffected.

3. **Widening `strategy_used` enum** — New strategy values (e.g. `layout_docling`, `vision_ocr`, `layout`) were added over time. The contract's `enum` clause captures 5 values. Adding new strategy names without removing old ones is COMPATIBLE — consumers that only check presence of the field continue to function.

---

### Q2: Confidence Scale Change Analysis

**Script output on `src/outputs/week3/extraction_ledger.jsonl`:**

```
min=0.650 max=0.900 mean=0.890 n=50
```

The `confidence` values in the ledger are in the range 0.0–1.0 (max=0.9), which is correct for the current scale. However:

- These are **top-level record confidences**, not `extracted_facts[].confidence` values. The nested facts do not exist in the original ledger.
- If a Week 4 consumer computed a threshold filter like `fact['confidence'] > 0.5`, it would succeed here — but if the scale were changed to 0–100, the filter would pass everything (all values 65–90 are > 0.5 on the wrong scale).
- The `ValidationRunner` range check (`minimum: 0.0, maximum: 1.0`) on `fact_confidence` is the exact check that would catch a 0–100 scale change with CRITICAL severity.

**Bitol YAML clause to prevent this:**
```yaml
schema:
  extracted_facts:
    type: array
    items:
      confidence:
        type: number
        minimum: 0.0
        maximum: 1.0
        description: >
          Confidence score for this extracted fact.
          MUST remain in 0.0–1.0 float range.
          BREAKING CHANGE if changed to 0–100 integer percentage.
        required: true
```

---

### Q3: Lineage-Based Attribution — Step by Step

Given a validation failure on `fact_confidence` (range check: max=51.3, expected <=1.0):

1. **Identify the source node** — look up `outputs/week3/extractions.jsonl` in the lineage graph nodes. Find its `node_id`.
2. **Traverse upstream edges** — query all edges where `target == node_id` and `relationship == PRODUCES` or `WRITES`. These are the nodes that wrote the data.
3. **For each upstream node** (source files in Week 3 system) — run `git log --follow -p <file>` to get all commits touching that file.
4. **Score each commit** — rank by: (a) keywords in commit message (`confidence`, `scale`, `percent`, `0-100`), (b) recency (commits after last known-good run ranked higher), (c) diff shows numeric literal changes near `confidence` field assignment.
5. **Build blast radius** — from the source node, traverse all downstream edges (`CONSUMES`, `READS`) to find every node that ingests `extracted_facts[]`. In the Week 4 snapshot, there are 3,024 `CONSUMES` edges and 610 `PRODUCES` edges across 2,307 nodes. The affected node list would include all TABLE and MODEL nodes downstream of the extraction pipeline.
6. **Emit violation log entry** — write `blame_chain[]` (top 3 scored commits) + `blast_radius{affected_nodes[]}` to `violation_log/violations.jsonl`.

---

### Q4: LangSmith Trace Contract (Bitol YAML)

```yaml
kind: DataContract
apiVersion: v3.0.0
id: langsmith-trace-runs
info:
  title: Langsmith Trace Runs
  version: 1.0.0
  owner: week7-enforcer
  description: 'Auto-generated contract for outputs/traces/runs.jsonl. Source snapshot:
    df705dc003f1620e. Generated at: 2026-04-01T17:47:03.308077+00:00.'
servers:
  local:
    type: local
    path: outputs/traces/runs.jsonl
    format: jsonl
terms:
  usage: Internal inter-system data contract. Do not publish.
  limitations: All confidence fields must remain in 0.0–1.0 float range. All _id fields
    must be valid UUID v4.
schema:
  id:
    type: string
    required: true
  name:
    type: string
    required: true
  run_type:
    type: string
    required: true
    enum:
    - chain
    - llm
    - tool
  error:
    type: string
    required: false
    enum:
    - evidence_not_found
  start_time:
    type: string
    required: true
  end_time:
    type: string
    required: true
  total_tokens:
    type: integer
    required: true
    minimum: 0
  prompt_tokens:
    type: integer
    required: true
    minimum: 0
  completion_tokens:
    type: integer
    required: true
    minimum: 0
  total_cost:
    type: number
    required: true
    minimum: 0
  parent_run_id:
    type: string
    required: false
    format: uuid
    pattern: ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$
  session_id:
    type: string
    required: true
    format: uuid
    pattern: ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$
quality:
  type: SodaChecks
  specification:
    checks for runs:
    - missing_count(id) = 0
    - missing_count(name) = 0
    - missing_count(run_type) = 0
    - missing_count(start_time) = 0
    - missing_count(end_time) = 0
    - missing_count(total_tokens) = 0
    - missing_count(prompt_tokens) = 0
    - missing_count(completion_tokens) = 0
    - missing_count(total_cost) = 0
    - missing_count(session_id) = 0
    - duplicate_count(session_id) = 0
lineage:
  upstream: []
  downstream:
  - id: transformation:/Users/melkam/Documents/10 Academy/Week4/ol-data-platform/src/ol_dbt/models/reporting/organization_administration_report.sql:1
    label: organization_administration_report.sql
    fields_consumed:
    - doc_id
    - extracted_facts
    - extraction_model
  - id: transformation:/Users/melkam/Documents/10 Academy/Week4/ol-data-platform/src/ol_dbt/models/reporting/mitxonline_course_engagements_daily_report.sql:1
    label: mitxonline_course_engagements_daily_report.sql
    fields_consumed:
    - doc_id
    - extracted_facts
    - extraction_model
  - id: transformation:/Users/melkam/Documents/10 Academy/Week4/ol-data-platform/src/ol_dbt/models/migration/edxorg_to_mitxonline_enrollments.sql:1
    label: edxorg_to_mitxonline_enrollments.sql
    fields_consumed:
    - doc_id
    - extracted_facts
    - extraction_model
  - id: transformation:/Users/melkam/Documents/10 Academy/Week4/ol-data-platform/src/ol_dbt/models/migration/edxorg_to_mitxonline_course_runs.sql:1
    label: edxorg_to_mitxonline_course_runs.sql
    fields_consumed:
    - doc_id
    - extracted_facts
    - extraction_model
  - id: transformation:/Users/melkam/Documents/10 Academy/Week4/ol-data-platform/src/ol_dbt/models/marts/mitxonline/marts__mitxonline_discussions.sql:1
    label: marts__mitxonline_discussions.sql
    fields_consumed:
    - doc_id
    - extracted_facts
    - extraction_model
  - id: transformation:/Users/melkam/Documents/10 Academy/Week4/ol-data-platform/src/ol_dbt/models/marts/mitxonline/marts__mitxonline_course_engagements_daily.sql:1
    label: marts__mitxonline_course_engagements_daily.sql
    fields_consumed:
    - doc_id
    - extracted_facts
    - extraction_model
  - id: transformation:/Users/melkam/Documents/10 Academy/Week4/ol-data-platform/src/ol_dbt/models/marts/mitxonline/marts__mitxonline_video_engagements.sql:1
    label: marts__mitxonline_video_engagements.sql
    fields_consumed:
    - doc_id
    - extracted_facts
    - extraction_model
  - id: transformation:/Users/melkam/Documents/10 Academy/Week4/ol-data-platform/src/ol_dbt/models/marts/mitxonline/marts__mitxonline_problem_summary.sql:1
    label: marts__mitxonline_problem_summary.sql
    fields_consumed:
    - doc_id
    - extracted_facts
    - extraction_model
  - id: transformation:/Users/melkam/Documents/10 Academy/Week4/ol-data-platform/src/ol_dbt/models/marts/mitxonline/marts__mitxonline_problem_submissions.sql:1
    label: marts__mitxonline_problem_submissions.sql
    fields_consumed:
    - doc_id
    - extracted_facts
    - extraction_model
  - id: transformation:/Users/melkam/Documents/10 Academy/Week4/ol-data-platform/src/ol_dbt/models/marts/combined/marts__combined__products.sql:1
    label: marts__combined__products.sql
    fields_consumed:
    - doc_id
    - extracted_facts
    - extraction_model
```

---

### Q5: Contract Staleness Failure Mode

**Most common production failure**: A contract is written against the initial schema, the schema evolves over months (new optional fields added, enum expanded, type loosened), and the contract is never updated. The contract still passes because the added flexibility means old checks still pass — but the contract no longer captures the real invariants. When a *new* breaking change is introduced (a field removed, a type tightened), the contract fails to catch it because the baseline has drifted.

**Concrete example from my platform**: The Week 3 extraction system ran 50 processing jobs with varying strategies. The "contract" was implicitly the first successful run's structure — `confidence` at top level, `doc_id` as a name string. By the time Week 4 consumed the output, the canonical schema had moved `confidence` into `extracted_facts[].confidence` and required UUID `doc_id`. The implicit contract never caught this because it was never written down.

**Architecture to prevent staleness**:
1. **ContractGenerator regenerates on every merge to main** — compares new auto-generated contract against committed contract YAML. CI fails if diff contains BREAKING changes without a version bump in `info.version`.
2. **SchemaEvolutionAnalyzer runs in CI** — stores a snapshot per contract per run in `schema_snapshots/`. Any BREAKING diff blocks merge.
3. **Baselines are versioned** — `schema_snapshots/baselines.json` is committed, not generated. Drift from baseline triggers WARN/FAIL in ValidationRunner. Current baselines seeded at `2026-04-01T08:09:44Z`.

---


## Confirmed Schema Deviations

See [data_flow_diagram.md](data_flow_diagram.md) for the full deviation table.

### Week 3 — `src/outputs/week3/extraction_ledger.jsonl`

| Issue | Canonical | Actual | Severity | Status |
|---|---|---|---|---|
| File path | `outputs/week3/extractions.jsonl` | `src/outputs/week3/extraction_ledger.jsonl` | BREAKING | ✅ Fixed by migration |
| `doc_id` type | UUID v4 | Document name string | BREAKING | ✅ Fixed (UUID5 deterministic) |
| `confidence` location | `extracted_facts[].confidence` | Top-level field | BREAKING | ✅ Fixed by migration |
| `extracted_facts[]` | Required array | Missing | BREAKING | ✅ Fixed (synthesised from block_count) |
| `entities[]` | Required array | Missing | BREAKING | ✅ Fixed (empty list — no entity data available) |
| `source_hash` | SHA-256 string | Missing | BREAKING | ✅ Fixed (sha256 of doc_id string) |
| `extracted_at` | ISO 8601 | Missing | BREAKING | ✅ Fixed (derived from processing_time offset) |

### Week 4 — `src/outputs/week4/lineage_graph.jsonl`

| Issue | Canonical | Actual | Severity | Status |
|---|---|---|---|---|
| File format | `.jsonl` (one snapshot per line) | Single JSON object | BREAKING | ✅ Fixed by migration |
| `snapshot_id` | UUID v4 | Missing | BREAKING | ✅ Fixed (uuid4 generated) |
| `git_commit` | 40-char hex | Missing | BREAKING | ✅ Fixed (`9d2b8c25...` from repo HEAD) |
| `captured_at` | ISO 8601 | Missing | BREAKING | ✅ Fixed (file mtime: `2026-03-31T14:53:14+00:00`) |
| `edges[].relationship` | Field named `relationship` | Field named `type` | BREAKING | ✅ Fixed by migration |
| `nodes[].type` values | `FILE\|TABLE\|SERVICE\|MODEL\|PIPELINE\|EXTERNAL` | `"transformation"`, `"dataset"` | BREAKING | ✅ Fixed (mapped to MODEL/TABLE) |
| `edges[].confidence` | Required float 0.0-1.0 | Missing | BREAKING | ✅ Fixed (set to 1.0 — asserted edges) |
| `node_id` format | UUID v4 | Composite path string | BREAKING | ⚠️ **Documented open violation** — path strings are the Cartographer's internal identity scheme; changing to UUID would break all edge references |

### Week 5 — `src/outputs/week5/events.json`

| Issue | Canonical | Actual | Severity | Status |
|---|---|---|---|---|
| File format | `.jsonl` | Single JSON array | BREAKING | ✅ Fixed by migration |
| `aggregate_id` | UUID v4 | Slug string (e.g. `loan-pg-doc-001`) | BREAKING | ✅ Fixed (UUID5 deterministic mapping, 270 IDs remapped) |
| `occurred_at` format | ISO 8601 with `T` separator | Postgres timestamp `2026-03-24 13:41:39+00` | BREAKING | ✅ Fixed (normalised to `2026-03-24T13:41:39+00:00`) |
| `recorded_at` format | ISO 8601 | Postgres timestamp | BREAKING | ✅ Fixed |
| `schema_version` | Semver `X.Y.Z` | Two-part `"1.0"` | BREAKING | ✅ Fixed (normalised to `"1.0.0"`) |

**Migration scripts required before implementation:**
- `outputs/migrate/week3_ledger_to_extractions.py` — transform `extraction_ledger.jsonl` to canonical schema ✅
- `outputs/migrate/week4_graph_to_snapshot.py` — transform `lineage_graph.json` to canonical `lineage_snapshots.jsonl` ✅
- `outputs/migrate/week5_events_to_jsonl.py` — transform `events.json` array to canonical `events.jsonl` ✅
