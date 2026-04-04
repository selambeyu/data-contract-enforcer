"""
ReportGenerator — Phase 5.

Auto-generates enforcer_report/report_data.json from live validation data.
Must be readable by someone who has never heard of a data contract.

Health score formula (spec-aligned):
    base = (checks_passed / total_checks) × 100
    deduct 20 for each CRITICAL violation
    deduct 10 for each HIGH violation

Required sections:
    1. Data Health Score (0–100) + one-sentence narrative
    2. Violations this week — count by severity, plain-language top 3
    3. Schema changes detected — plain-language summary with compatibility verdict
    4. AI system risk assessment — drift, prompt schema, output violation rate
    5. Recommended actions — 3 specific, ordered by risk reduction value

LLM Narration (optional — enriches narrative, violations, and recommendations):
    Priority order:
      1. OpenAI   — set OPENAI_API_KEY  (model: gpt-4o-mini)
      2. Ollama   — set OLLAMA_HOST (default http://localhost:11434)
                    set OLLAMA_MODEL (default llama3.2)
      3. Fallback — heuristic strings (no LLM required)

CLI:
    python src/contracts/report_generator.py \
        --validation-dir validation_reports \
        --violations violation_log/violations.jsonl \
        --diffs-dir validation_reports \
        --ai-extensions enforcer_report/ai_extensions.json \
        --registry contract_registry/subscriptions.yaml \
        --output enforcer_report/report_data.json
"""

import argparse
import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file: src/contracts/ → project root)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ── LLM Narrator ──────────────────────────────────────────────────────────────

class LLMNarrator:
    """
    Thin LLM wrapper used exclusively for generating human-readable report text.

    Priority (first available wins):
      1. OpenAI      — OPENAI_API_KEY set                  (model: gpt-4o-mini)
      2. OpenRouter  — OPENROUTER_API_KEY set              (model: OPENROUTER_MODEL, default openai/gpt-4o-mini)
      3. Ollama      — OLLAMA_HOST reachable               (model: OLLAMA_MODEL, default llama3.2:3b)
      4. None        — caller falls back to heuristic strings

    All keys are loaded from .env via python-dotenv before this class is instantiated.

    Usage:
        narrator = LLMNarrator()
        text = narrator.complete(prompt)   # returns str or None
    """

    _SYSTEM_PROMPT = (
        "You are a senior data engineer writing a plain-English data quality report. "
        "Be concise, specific, and avoid jargon. Do not use bullet points unless asked. "
        "Output only the requested text — no preamble, no markdown headers."
    )

    _OPENAI_MODEL        = "gpt-4o-mini"
    _OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    _OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"
    _OLLAMA_DEFAULT_HOST  = "http://localhost:11434"
    _OLLAMA_DEFAULT_MODEL = "llama3.2:3b"

    def __init__(self) -> None:
        self._backend: str | None = None

        # Read all config from environment (populated by load_dotenv at module top)
        self._openai_key       = os.environ.get("OPENAI_API_KEY", "").strip()
        self._openrouter_key   = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self._openrouter_model = os.environ.get("OPENROUTER_MODEL", self._OPENROUTER_DEFAULT_MODEL)
        self._ollama_host      = os.environ.get("OLLAMA_HOST", self._OLLAMA_DEFAULT_HOST).rstrip("/")
        self._ollama_model     = os.environ.get("OLLAMA_MODEL", self._OLLAMA_DEFAULT_MODEL)

        if self._openai_key:
            self._backend = "openai"
            print(f"[llm] Backend: OpenAI (model: {self._OPENAI_MODEL})")
        elif self._openrouter_key:
            self._backend = "openrouter"
            print(f"[llm] Backend: OpenRouter (model: {self._openrouter_model})")
        elif self._ollama_reachable():
            self._backend = "ollama"
            print(f"[llm] Backend: Ollama at {self._ollama_host} (model: {self._ollama_model})")
        else:
            print(
                "[llm] No LLM backend available — falling back to heuristic narration.\n"
                "      Set OPENAI_API_KEY, OPENROUTER_API_KEY, or start Ollama locally."
            )

    # ── Backend detection ──────────────────────────────────────────────────────

    def _ollama_reachable(self) -> bool:
        try:
            req = urllib.request.Request(f"{self._ollama_host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3):
                return True
        except Exception:
            return False

    # ── Public interface ───────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._backend is not None

    def complete(self, prompt: str, max_tokens: int = 400) -> str | None:
        """Call the active backend. Returns plain text string or None on failure."""
        if self._backend == "openai":
            return self._openai_complete(prompt, max_tokens)
        if self._backend == "openrouter":
            return self._openrouter_complete(prompt, max_tokens)
        if self._backend == "ollama":
            return self._ollama_complete(prompt, max_tokens)
        return None

    # ── Backend implementations ────────────────────────────────────────────────

    def _openai_complete(self, prompt: str, max_tokens: int) -> str | None:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._openai_key)
            resp = client.chat.completions.create(
                model=self._OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[llm] OpenAI error: {exc}")
            return None

    def _openrouter_complete(self, prompt: str, max_tokens: int) -> str | None:
        """
        OpenRouter is OpenAI-API-compatible — uses the same SDK with a custom base_url.
        Supports hundreds of models (OpenAI, Anthropic, Mistral, etc.) via one key.
        See https://openrouter.ai/models for available model IDs.
        """
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self._openrouter_key,
                base_url=self._OPENROUTER_BASE_URL,
            )
            resp = client.chat.completions.create(
                model=self._openrouter_model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[llm] OpenRouter error: {exc}")
            return None

    def _ollama_complete(self, prompt: str, max_tokens: int) -> str | None:
        try:
            payload = json.dumps({
                "model": self._ollama_model,
                "prompt": self._SYSTEM_PROMPT + "\n\n" + prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.3},
            }).encode()
            req = urllib.request.Request(
                f"{self._ollama_host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                return data.get("response", "").strip()
        except Exception as exc:
            print(f"[llm] Ollama error: {exc}")
            return None


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_validation_reports(val_dir: Path) -> list[dict]:
    reports = []
    for p in sorted(val_dir.glob("*.json")):
        try:
            with open(p) as f:
                reports.append(json.load(f))
        except Exception:
            pass
    return reports


def load_violations(violations_path: Path) -> list[dict]:
    if not violations_path.exists():
        return []
    entries = []
    with open(violations_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries


def load_diffs(diffs_dir: Path) -> list[dict]:
    diffs = []
    for p in sorted(diffs_dir.glob("schema_evolution_*.json")):
        try:
            with open(p) as f:
                diffs.append(json.load(f))
        except Exception:
            pass
    return diffs


def load_ai_extensions(ai_path: Path) -> dict:
    if not ai_path.exists():
        return {}
    with open(ai_path) as f:
        return json.load(f)


def load_registry(registry_path: Path) -> list[dict]:
    if not registry_path.exists():
        return []
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    return data.get("subscriptions", [])


# ── LLM narration helpers ─────────────────────────────────────────────────────

def llm_health_narrative(
    narrator: "LLMNarrator",
    score: int,
    grade: str,
    total_checks: int,
    total_passed: int,
    total_critical: int,
    total_high: int,
    breaking_changes: int,
    contracts: list[str],
) -> str | None:
    if not narrator.available:
        return None
    prompt = f"""
Write a 2-3 sentence plain-English executive summary for a data quality report.

Facts:
- Data Health Score: {score}/100 (Grade {grade})
- Contract checks run: {total_checks} across {len(contracts)} contracts ({', '.join(contracts)})
- Checks passed: {total_passed}/{total_checks}
- CRITICAL violations: {total_critical}
- HIGH violations: {total_high}
- Breaking schema changes detected: {breaking_changes}

The summary should explain what the score means, which systems are affected, and the urgency level.
Do not repeat the numbers verbatim — explain their meaning in plain language.
""".strip()
    return narrator.complete(prompt, max_tokens=180)


def llm_plain_violation(narrator: "LLMNarrator", violation: dict) -> str | None:
    if not narrator.available:
        return None
    cid      = violation.get("contract_id", "unknown")
    check_id = violation.get("check_id", "")
    severity = violation.get("severity", "")
    fc       = violation.get("failing_check", {})
    msg      = fc.get("message", "")
    records  = fc.get("records_failing", 0)
    br       = violation.get("blast_radius", {})
    nodes    = [n.get("node_id", "") for n in br.get("affected_nodes", [])]
    pipelines = br.get("affected_pipelines", [])
    blame    = violation.get("blame_chain", [{}])
    top_file = blame[0].get("file_path", "unknown") if blame else "unknown"
    prompt = f"""
Write one plain-English sentence describing this data contract violation for a non-technical stakeholder.
Include: what broke, which system it affects, and the business impact.

Details:
- Severity: {severity}
- Contract: {cid}
- Check: {check_id}
- What failed: {msg}
- Records affected: {records}
- Downstream systems at risk: {', '.join(nodes[:3]) or 'none'}
- Affected pipelines: {', '.join(pipelines[:3]) or 'none'}
- Likely source file: {top_file}

One sentence only. Start with the severity level in brackets like [{severity}].
""".strip()
    return narrator.complete(prompt, max_tokens=120)


def llm_recommendations(
    narrator: "LLMNarrator",
    violations: list[dict],
    diffs: list[dict],
    ai_results: dict,
    heuristic_recs: list[str],
) -> list[str] | None:
    if not narrator.available:
        return None

    # Build a compact context block
    crit_viols = [v for v in violations if v.get("severity") == "CRITICAL"]
    high_viols = [v for v in violations if v.get("severity") == "HIGH"]
    breaking   = [d for d in diffs if d.get("summary", {}).get("breaking", 0) > 0]
    drift_warn = ai_results.get("embedding_drift", {}).get("status") == "WARN"
    output_warn = ai_results.get("llm_output_violation_rate", {}).get("status") == "WARN"

    issues = []
    for v in crit_viols[:2]:
        issues.append(f"CRITICAL: {v.get('check_id')} — {v.get('failing_check', {}).get('message', '')}")
    for v in high_viols[:2]:
        issues.append(f"HIGH: {v.get('check_id')} — {v.get('failing_check', {}).get('message', '')}")
    for d in breaking[:2]:
        cid = d.get("contract_id", "?")
        n   = d.get("summary", {}).get("breaking", 0)
        issues.append(f"BREAKING schema change in {cid}: {n} field(s) removed or type-changed")
    if drift_warn:
        issues.append("WARNING: Embedding drift detected in extracted text features")
    if output_warn:
        issues.append("WARNING: LLM output schema violation rate above threshold")

    prompt = f"""
You are a senior data engineer. Based on the issues below, write exactly 3 prioritised recommendations.
Each recommendation must:
- Name the specific contract, field, or script involved (use the check_id or contract name directly)
- State the concrete action to take
- Be one sentence long

Issues:
{chr(10).join(f'  - {i}' for i in issues) if issues else '  - No active violations'}

Heuristic recommendations already computed (improve on these, do not just repeat them):
{chr(10).join(f'  {i+1}. {r[:200]}' for i, r in enumerate(heuristic_recs))}

Output exactly 3 lines, each starting with the priority number like "1. ..." "2. ..." "3. ..."
""".strip()

    result = narrator.complete(prompt, max_tokens=350)
    if not result:
        return None
    # Parse "1. ...\n2. ...\n3. ..." into a list
    lines = [ln.strip() for ln in result.splitlines() if ln.strip()]
    recs = []
    for ln in lines:
        # strip leading "1. " "2. " etc.
        import re
        cleaned = re.sub(r"^\d+\.\s*", "", ln)
        if cleaned:
            recs.append(cleaned)
    return recs[:3] if recs else None


# ── Health score (spec formula) ────────────────────────────────────────────────

def compute_data_health_score(
    validation_reports: list[dict],
    ai_results: dict,
) -> tuple[int, list[str]]:
    """
    Spec formula:
        base = (checks_passed / total_checks) × 100
        deduct 20 for each CRITICAL violation
        deduct 10 for each HIGH violation
        cap at 0 (never negative)

    Only baseline reports (not violated_ datasets) count toward the score.
    """
    # Only count clean/baseline validation runs
    baseline_reports = [
        r for r in validation_reports
        if not Path(r.get("data_path", "")).name.startswith("violated")
        and not Path(r.get("data_path", "")).name.endswith("_violated.jsonl")
    ]

    total_checks = sum(r.get("total_checks", 0) for r in baseline_reports)
    total_passed = sum(r.get("passed", 0) for r in baseline_reports)
    total_critical = sum(r.get("critical", 0) for r in baseline_reports)
    total_high = sum(r.get("high", 0) for r in baseline_reports)

    if total_checks == 0:
        base = 100.0
    else:
        base = (total_passed / total_checks) * 100

    deductions = []
    if total_critical > 0:
        deductions.append(f"{total_critical} CRITICAL violation(s) on baseline data (−{total_critical * 20} points)")
        base -= total_critical * 20
    if total_high > 0:
        deductions.append(f"{total_high} HIGH violation(s) on baseline data (−{total_high * 10} points)")
        base -= total_high * 10

    # AI extension deductions (minor)
    for key, val in ai_results.items():
        if isinstance(val, dict) and val.get("status") == "FAIL":
            deductions.append(f"AI extension '{key}' FAIL (−5 points)")
            base -= 5

    score = max(0, int(base))
    return score, deductions


# ── Plain English violations ───────────────────────────────────────────────────

def plain_english_violations(violations: list[dict]) -> list[str]:
    """
    Spec: each description must name the failing system, the failing field,
    and the impact on downstream consumers.
    """
    summaries = []
    for v in violations:
        cid = v.get("contract_id", "unknown")
        check_id = v.get("check_id", "")
        severity = v.get("severity", "")
        fc = v.get("failing_check", {})
        msg = fc.get("message", "")
        records_failing = fc.get("records_failing", 0)
        br = v.get("blast_radius", {})
        affected = br.get("affected_nodes", [])
        subscriber_names = [n.get("node_id", "") for n in affected]
        pipelines = br.get("affected_pipelines", [])

        summaries.append(
            f"[{severity}] Contract '{cid}': {msg} "
            f"({records_failing} records affected). "
            + (
                f"Downstream consumers at risk: {', '.join(subscriber_names[:3])}. "
                if subscriber_names else
                "No registered subscribers directly affected. "
            )
            + (f"Affected pipelines: {', '.join(pipelines[:3])}." if pipelines else "")
        )
    return summaries


# ── Recommendations ────────────────────────────────────────────────────────────

def generate_recommendations(
    validation_reports: list[dict],
    violations: list[dict],
    diffs: list[dict],
    ai_results: dict,
    subscriptions: list[dict],
) -> list[str]:
    """
    Spec: 3 specific prioritised actions ordered by risk reduction value.
    Each must name the specific file, field, and contract clause — not generic advice.
    """
    candidates: list[tuple[int, str]] = []  # (priority, text)

    # Check for HIGH range violations on baseline data (confidence scale)
    for r in validation_reports:
        if Path(r.get("data_path", "")).name.startswith("violated"):
            continue
        for result in r.get("results", []):
            if result.get("status") == "HIGH" and result.get("check_type") == "range":
                cid = r.get("contract_id", "?")
                col = result.get("column_name", "?")
                actual = result.get("actual_value", "?")
                expected = result.get("expected", "?")
                candidates.append((1,
                    f"Fix range violation in '{cid}' field '{col}': actual={actual}, "
                    f"expected={expected}. "
                    f"Update the producer to output values within the contracted range. "
                    f"This is likely a scale change (0.0–1.0 → 0–100). "
                    f"Fix in the migration script under outputs/migrate/ before next deployment."
                ))

    # Check for CRITICAL violations in baseline reports
    for r in validation_reports:
        if Path(r.get("data_path", "")).name.startswith("violated"):
            continue
        if r.get("critical", 0) > 0:
            cid = r.get("contract_id", "?")
            for result in r.get("results", []):
                if result.get("status") == "CRITICAL":
                    col = result.get("column_name", "?")
                    msg = result.get("message", "?")
                    candidates.append((1,
                        f"Fix CRITICAL violation in '{cid}' on field '{col}': {msg} "
                        f"Inspect outputs/migrate/ scripts and the canonical schema in generated_contracts/{cid}.yaml."
                    ))
                    break

    # Check for breaking schema changes
    for diff in diffs:
        if diff.get("summary", {}).get("breaking", 0) > 0:
            cid = diff.get("contract_id", "?")
            breaking_changes = [c for c in diff.get("changes", []) if c.get("classification") == "BREAKING"]
            removed_fields = [c["field"] for c in breaking_changes if c.get("change_type") == "field_removed"]
            if removed_fields:
                candidates.append((2,
                    f"Bump version for contract '{cid}' before deploying: "
                    f"{len(removed_fields)} field(s) removed: {', '.join(removed_fields)}. "
                    f"Notify all registry subscribers (see contract_registry/subscriptions.yaml). "
                    f"Minimum two-sprint deprecation period before removing any field."
                ))
            elif breaking_changes:
                ct = breaking_changes[0].get("change_type", "change")
                field = breaking_changes[0].get("field", "?")
                candidates.append((2,
                    f"Breaking schema change in '{cid}': {ct} on field '{field}'. "
                    f"Bump version and distribute migration_impact report to all subscribers "
                    f"before deploying. See validation_reports/schema_evolution_{cid}.json."
                ))

    # Check for subscribers in ENFORCE mode that would block
    enforce_subs = [
        s for s in subscriptions if s.get("validation_mode") == "ENFORCE"
    ]
    if enforce_subs:
        cids_enforce = list({s["contract_id"] for s in enforce_subs})
        candidates.append((3,
            f"Run ValidationRunner in ENFORCE mode for production deployments of: "
            f"{', '.join(cids_enforce[:3])}. "
            f"These contracts have subscribers registered with validation_mode=ENFORCE — "
            f"any CRITICAL or HIGH violation will block the pipeline. "
            f"Start with AUDIT mode on any new dataset to calibrate thresholds first."
        ))

    # AI drift
    drift = ai_results.get("embedding_drift", {})
    if drift.get("status") == "WARN":
        drift_score = drift.get("drift_score", "?")
        method = drift.get("embedding_method", "tfidf_fallback")
        candidates.append((3,
            f"Investigate embedding drift in extracted_facts (drift_score={drift_score}, method={method}). "
            f"If using TF-IDF fallback, set OPENAI_API_KEY to enable real semantic drift detection. "
            f"Real embeddings use text-embedding-3-small with threshold=0.15 cosine distance."
        ))

    # LLM output violation rate rising
    out_rate = ai_results.get("llm_output_violation_rate", {})
    if out_rate.get("trend") == "rising" or out_rate.get("status") == "WARN":
        rate = out_rate.get("violation_rate", 0)
        candidates.append((2,
            f"LLM output schema violation rate is {rate*100:.1f}% "
            f"(threshold=2.0%). "
            f"Check overall_verdict values in outputs/week2/verdicts.jsonl — "
            f"any value outside {{PASS, FAIL, WARN}} indicates prompt template degradation. "
            f"Review recent prompt changes and re-run ai_extensions.py to track trend."
        ))

    candidates.sort(key=lambda x: x[0])
    return [text for _, text in candidates[:3]]


# ── Per-contract summary table ─────────────────────────────────────────────────

def build_contract_summary_table(
    validation_reports: list[dict],
    diffs: list[dict],
    subscriptions: list[dict],
) -> list[dict]:
    baseline_by_cid = {}
    for r in validation_reports:
        cid = r.get("contract_id", "")
        if cid and not Path(r.get("data_path", "")).name.startswith("violated"):
            baseline_by_cid[cid] = r

    diff_by_cid = {d.get("contract_id"): d for d in diffs}
    all_cids = set(baseline_by_cid) | set(diff_by_cid)

    rows = []
    for cid in sorted(all_cids):
        r    = baseline_by_cid.get(cid, {})
        diff = diff_by_cid.get(cid, {})
        subs = [s for s in subscriptions if s.get("contract_id") == cid]
        tier1 = sum(1 for s in subs if s.get("tier", 1) < 3)
        tier3 = len(subs) - tier1
        enforce_mode = sum(1 for s in subs if s.get("validation_mode") == "ENFORCE")
        rows.append({
            "contract_id":           cid,
            "total_checks":          r.get("total_checks", "—"),
            "passed":                r.get("passed", "—"),
            "high":                  r.get("high", "—"),
            "critical":              r.get("critical", "—"),
            "schema_breaking":       diff.get("summary", {}).get("breaking", "—"),
            "schema_compatible":     diff.get("summary", {}).get("compatible", "—"),
            "subscriber_count":      len(subs),
            "tier1_subscribers":     tier1,
            "tier3_subscribers":     tier3,
            "enforce_mode_subs":     enforce_mode,
            "requires_version_bump": diff.get("summary", {}).get("requires_version_bump", False),
            "enforcement_mode":      r.get("enforcement_mode", "AUDIT"),
        })
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate enforcer summary report.")
    parser.add_argument("--validation-dir", default="validation_reports")
    parser.add_argument("--violations",     default="violation_log/violations.jsonl")
    parser.add_argument("--diffs-dir",      default="validation_reports")
    parser.add_argument("--ai-extensions",  default="enforcer_report/ai_extensions.json")
    parser.add_argument("--registry",       default="contract_registry/subscriptions.yaml")
    parser.add_argument("--output",         default="enforcer_report/report_data.json")
    args = parser.parse_args()

    print("[report_generator] Loading data ...")
    validation_reports = load_validation_reports(Path(args.validation_dir))
    violations         = load_violations(Path(args.violations))
    diffs              = load_diffs(Path(args.diffs_dir))
    ai_results         = load_ai_extensions(Path(args.ai_extensions))
    subscriptions      = load_registry(Path(args.registry))

    print(f"  Validation reports : {len(validation_reports)}")
    print(f"  Violation entries  : {len(violations)}")
    print(f"  Schema diffs       : {len(diffs)}")
    print(f"  Subscriptions      : {len(subscriptions)}")

    # Initialise LLM narrator (auto-detects OpenAI → Ollama → heuristic)
    narrator = LLMNarrator()

    health_score, deductions = compute_data_health_score(validation_reports, ai_results)
    heuristic_recs   = generate_recommendations(
        validation_reports, violations, diffs, ai_results, subscriptions
    )
    contract_table   = build_contract_summary_table(validation_reports, diffs, subscriptions)

    # Total counts (baseline only)
    baseline_reports = [
        r for r in validation_reports
        if not Path(r.get("data_path", "")).name.startswith("violated")
    ]
    total_checks   = sum(r.get("total_checks", 0) for r in baseline_reports)
    total_passed   = sum(r.get("passed", 0)        for r in baseline_reports)
    total_critical = sum(r.get("critical", 0)      for r in baseline_reports)
    total_high     = sum(r.get("high", 0)           for r in baseline_reports)
    breaking_changes = sum(d.get("summary", {}).get("breaking", 0) for d in diffs)
    grade = (
        "A" if health_score >= 90 else
        "B" if health_score >= 80 else
        "C" if health_score >= 70 else
        "D" if health_score >= 60 else "F"
    )
    contract_names = [r.get("contract_id", "") for r in baseline_reports if r.get("contract_id")]

    # ── LLM-enriched narrative (falls back gracefully) ─────────────────────────
    print("[report_generator] Generating narrative ...")

    health_narrative = llm_health_narrative(
        narrator, health_score, grade,
        total_checks, total_passed, total_critical, total_high,
        breaking_changes, list(dict.fromkeys(contract_names)),
    ) or (
        f"The monitored data platform has {total_checks} contract checks across "
        f"{len(baseline_reports)} baseline validation runs. "
        f"{total_passed}/{total_checks} checks pass. "
        + (f"{total_critical} CRITICAL and {total_high} HIGH violations detected. "
           if (total_critical + total_high) else "No critical violations on baseline data. ")
        + f"Score: {health_score}/100."
    )

    # ── LLM-enriched violation descriptions ───────────────────────────────────
    plain_violations_heuristic = plain_english_violations(violations)
    if narrator.available:
        plain_violations: list[str] = []
        for v in violations[:3]:
            desc = llm_plain_violation(narrator, v) or plain_violations_heuristic[len(plain_violations)] \
                if len(plain_violations) < len(plain_violations_heuristic) else ""
            plain_violations.append(desc)
    else:
        plain_violations = plain_violations_heuristic[:3]

    # ── LLM-enriched recommendations ──────────────────────────────────────────
    recommendations = llm_recommendations(
        narrator, violations, diffs, ai_results, heuristic_recs
    ) or heuristic_recs

    # ── Assemble report ────────────────────────────────────────────────────────
    report = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "llm_backend":       narrator._backend or "heuristic",
        "data_health_score": health_score,
        "health_grade":      grade,
        "health_narrative":  health_narrative,
        # Section 2
        "violations_this_week": {
            "critical_count": total_critical,
            "high_count":     total_high,
            "total_logged":   len(violations),
            "plain_english":  plain_violations,
        },
        # Section 3
        "schema_changes_detected": {
            "total_diffs_run":  len(diffs),
            "breaking_changes": breaking_changes,
            "summaries": [
                {
                    "contract_id": d.get("contract_id"),
                    "verdict":     d.get("summary", {}).get("compatibility_verdict", "UNKNOWN"),
                    "breaking":    d.get("summary", {}).get("breaking", 0),
                    "compatible":  d.get("summary", {}).get("compatible", 0),
                    "action_required": (
                        "Bump version and notify registry subscribers."
                        if d.get("summary", {}).get("requires_version_bump")
                        else "No action required."
                    ),
                }
                for d in diffs
            ],
        },
        # Section 4
        "ai_system_risk_assessment": {
            k: {"status": v.get("status"), "message": v.get("message")}
            for k, v in ai_results.items()
            if isinstance(v, dict)
        },
        # Section 5
        "top_recommendations": recommendations,
        # Supporting data
        "score_deductions":       deductions,
        "contract_summary_table": contract_table,
        "totals": {
            "contracts_covered":       len(contract_table),
            "total_subscriptions":     len(subscriptions),
            "total_checks_baseline":   total_checks,
            "total_violations_logged": len(violations),
            "breaking_schema_changes": breaking_changes,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[report_generator] ── Enforcer Report ──────────────────────")
    print(f"  LLM backend       : {report['llm_backend']}")
    print(f"  Data Health Score : {health_score}/100 (Grade: {grade})")
    print(f"  {health_narrative}")
    print(f"\n  Top Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"    {i}. {rec[:130]}")
    print(f"\n[report_generator] Written: {output_path}")


if __name__ == "__main__":
    main()
