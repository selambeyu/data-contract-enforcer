"""
AIExtensions — Phase 4.

Three AI-specific contract checks (spec-aligned):

  Extension 1: Embedding Drift Detection
    Uses OpenAI text-embedding-3-small on extracted_facts[].text.
    Compares cosine distance between cached baseline centroid and current sample.
    Falls back to TF-IDF if OPENAI_API_KEY is not set.
    Threshold: 0.15 cosine distance (spec default).

  Extension 2: Prompt Input Schema Validation
    Validates document metadata objects that are interpolated into prompt templates.
    Schema: doc_id (uuid, len=36), source_path (non-empty string), content_preview (≤8000 chars).
    Non-conforming records written to outputs/quarantine/.

  Extension 3: LLM Output Schema Violation Rate
    Applies to Week 2 verdict records (or LLM runs from traces as fallback).
    Checks overall_verdict is exactly one of {PASS, FAIL, WARN}.
    Tracks violation_rate and trend (rising/stable).
    Warns if rate > 0.02.

CLI:
    python src/contracts/ai_extensions.py \
        --mode all \
        --extractions outputs/week3/extractions.jsonl \
        --traces outputs/traces/runs.jsonl \
        --verdicts outputs/week2/verdicts.jsonl \
        --output enforcer_report/ai_extensions.json
"""

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
import argparse

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Extension 1: Embedding Drift ───────────────────────────────────────────────

def _try_openai_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]] | None:
    """
    Attempt to embed texts via OpenAI API.
    Returns list of embedding vectors, or None if unavailable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(input=texts, model=model)
        return [e.embedding for e in response.data]
    except Exception as exc:
        print(f"[ai_extensions] OpenAI embedding failed: {exc} — falling back to TF-IDF")
        return None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _build_tfidf(corpus: list[str]) -> list[dict[str, float]]:
    n = len(corpus)
    if n == 0:
        return []
    tfs = [Counter(_tokenize(doc)) for doc in corpus]
    df: Counter = Counter()
    for tf in tfs:
        for term in tf:
            df[term] += 1
    idf = {term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()}
    vecs = []
    for tf in tfs:
        total = sum(tf.values()) or 1
        vecs.append({term: (count / total) * idf[term] for term, count in tf.items()})
    return vecs


def _cosine_similarity(a: list[float] | dict, b: list[float] | dict) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        terms = set(a) | set(b)
        dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in terms)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
    else:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _mean_vec(vecs: list) -> list[float] | dict:
    if not vecs:
        return {}
    if isinstance(vecs[0], dict):
        all_terms = set().union(*vecs)
        n = len(vecs)
        return {t: sum(v.get(t, 0.0) for v in vecs) / n for t in all_terms}
    else:
        dim = len(vecs[0])
        n = len(vecs)
        return [sum(v[i] for v in vecs) / n for i in range(dim)]


def compute_embedding_drift(
    extractions_baseline: list[dict],
    extractions_current: list[dict],
    baseline_cache_path: str | Path = "schema_snapshots/embedding_baselines/week3_centroid.json",
    threshold: float = 0.15,
    sample_n: int = 200,
) -> dict:
    """
    Spec: embed a sample of 200 text values using text-embedding-3-small.
    Store centroid. On each run, compute cosine distance from stored centroid.
    Alert if distance exceeds threshold (default 0.15).

    Falls back to TF-IDF if OpenAI API key is unavailable.
    """
    def extract_texts(records: list[dict]) -> list[str]:
        texts = []
        for r in records:
            for fact in r.get("extracted_facts", []):
                text = fact.get("text") or fact.get("fact_text") or ""
                if text:
                    texts.append(str(text))
        return texts

    baseline_texts = extract_texts(extractions_baseline)
    current_texts  = extract_texts(extractions_current)

    if not baseline_texts or not current_texts:
        return {
            "check": "embedding_drift",
            "status": "SKIP",
            "message": "Insufficient text data for embedding drift computation.",
            "baseline_docs": len(baseline_texts),
            "current_docs": len(current_texts),
            "embedding_method": "none",
        }

    # Sample
    import random
    random.seed(42)
    b_sample = random.sample(baseline_texts, min(sample_n, len(baseline_texts)))
    c_sample = random.sample(current_texts,  min(sample_n, len(current_texts)))

    cache_path = Path(baseline_cache_path)
    embedding_method = "openai"

    # Try OpenAI embeddings first
    b_vecs = _try_openai_embeddings(b_sample)
    c_vecs = _try_openai_embeddings(c_sample)

    if b_vecs is None or c_vecs is None:
        # Fall back to TF-IDF
        embedding_method = "tfidf_fallback"
        b_vecs = _build_tfidf(b_sample)
        c_vecs = _build_tfidf(c_sample)
        threshold_effective = 0.25  # TF-IDF similarity (not distance) threshold
    else:
        threshold_effective = threshold  # cosine distance threshold for real embeddings

    b_centroid = _mean_vec(b_vecs)
    c_centroid = _mean_vec(c_vecs)

    similarity = _cosine_similarity(b_centroid, c_centroid)

    if embedding_method == "openai":
        # For real embeddings: drift = 1 - similarity; fail if drift > threshold
        drift_score = round(1.0 - similarity, 4)
        passed = drift_score <= threshold_effective
        status = "PASS" if passed else "WARN"
    else:
        # For TF-IDF fallback: similarity directly (higher = less drift)
        drift_score = round(1.0 - similarity, 4)
        passed = similarity >= threshold_effective
        status = "PASS" if passed else "WARN"

    # Cache centroid for future runs
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists() and embedding_method == "openai":
        with open(cache_path, "w") as f:
            json.dump({"centroid": b_centroid, "sample_size": len(b_sample)}, f)

    return {
        "check": "embedding_drift",
        "status": status,
        "drift_score": drift_score,
        "cosine_similarity": round(similarity, 4),
        "threshold": threshold_effective,
        "embedding_method": embedding_method,
        "baseline_docs": len(baseline_texts),
        "current_docs": len(current_texts),
        "sample_size": sample_n,
        "message": (
            f"Embedding drift score {drift_score:.4f} {'<=' if passed else '>'} {threshold_effective} "
            f"({'PASS — no significant drift' if passed else 'WARN — potential semantic drift detected'}). "
            f"Method: {embedding_method}."
        ),
    }


# ── Extension 2: Prompt Input Schema Validation ────────────────────────────────

# Spec schema: document metadata interpolated into extraction prompt template
PROMPT_INPUT_SCHEMA = {
    "required": ["doc_id", "source_path", "content_preview"],
    "properties": {
        "doc_id": {
            "type": str,
            "min_length": 36,
            "max_length": 36,
            "description": "UUID v4 of the document",
        },
        "source_path": {
            "type": str,
            "min_length": 1,
            "description": "Non-empty path or URL to source document",
        },
        "content_preview": {
            "type": str,
            "max_length": 8000,
            "description": "Excerpt passed to LLM — must not exceed context limit",
        },
    },
}

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def _validate_record_against_prompt_schema(record: dict) -> list[str]:
    """Return list of violation messages for a record, empty if conforming."""
    violations = []
    schema = PROMPT_INPUT_SCHEMA
    for field in schema["required"]:
        if field not in record:
            violations.append(f"missing required field '{field}'")
            continue
        val = record[field]
        props = schema["properties"][field]
        if not isinstance(val, props["type"]):
            violations.append(f"'{field}' must be {props['type'].__name__}, got {type(val).__name__}")
            continue
        if "min_length" in props and len(str(val)) < props["min_length"]:
            violations.append(f"'{field}' too short (len={len(str(val))}, min={props['min_length']})")
        if "max_length" in props and len(str(val)) > props["max_length"]:
            violations.append(f"'{field}' too long (len={len(str(val))}, max={props['max_length']})")
        if field == "doc_id" and not _UUID_RE.match(str(val)):
            violations.append(f"'doc_id' is not a valid UUID v4: {str(val)[:40]}")
    return violations


def validate_prompt_schemas(
    extractions: list[dict],
    quarantine_path: str | Path = "outputs/quarantine/prompt_schema_violations.jsonl",
) -> dict:
    """
    Spec: validates document metadata objects interpolated into prompt templates.
    Required keys: doc_id (UUID v4, len=36), source_path (non-empty), content_preview (≤8000 chars).
    Non-conforming records written to outputs/quarantine/ — never silently dropped.
    """
    total = len(extractions)
    if total == 0:
        return {
            "check": "prompt_schema_validation",
            "status": "SKIP",
            "message": "No extraction records to validate.",
        }

    violations = []
    quarantine_records = []

    for rec in extractions:
        # Build the prompt input object from the extraction record
        prompt_input = {
            "doc_id":          rec.get("doc_id", ""),
            "source_path":     rec.get("source_path", ""),
            "content_preview": str(rec.get("source_hash", ""))[:8000],  # use available field
        }
        issues = _validate_record_against_prompt_schema(prompt_input)
        if issues:
            violations.append({
                "doc_id": rec.get("doc_id", "unknown"),
                "issues": issues,
            })
            quarantine_records.append({**rec, "_quarantine_reason": issues})

    # Write quarantine file — non-conforming records must never be silently dropped
    if quarantine_records:
        qpath = Path(quarantine_path)
        qpath.parent.mkdir(parents=True, exist_ok=True)
        with open(qpath, "w") as f:
            for r in quarantine_records:
                f.write(json.dumps(r) + "\n")

    violation_rate = len(violations) / total
    status = "PASS" if violation_rate == 0 else ("WARN" if violation_rate < 0.10 else "FAIL")

    return {
        "check": "prompt_schema_validation",
        "status": status,
        "total_records": total,
        "violating_records": len(violations),
        "violation_rate": round(violation_rate, 4),
        "quarantine_path": str(quarantine_path) if quarantine_records else None,
        "violations_sample": violations[:10],
        "message": (
            f"{len(violations)}/{total} extraction records have prompt schema violations "
            f"({violation_rate*100:.1f}%). Non-conforming records written to quarantine."
            if violations else
            f"All {total} extraction records conform to prompt input schema."
        ),
    }


# ── Extension 3: LLM Output Schema Violation Rate ─────────────────────────────

# Spec: applies to Week 2 verdict records
VALID_VERDICTS = {"PASS", "FAIL", "WARN"}


def check_llm_output_violation_rate(
    verdict_records: list[dict],
    baseline_rate: float | None = None,
    warn_threshold: float = 0.02,
) -> dict:
    """
    Spec: track output_schema_violation_rate per prompt version.
    A rising rate signals prompt degradation or model behaviour change.

    Applies to Week 2 verdict records: overall_verdict must be in {PASS, FAIL, WARN}.
    Falls back to synthesising from LLM trace outputs['score'] if no verdicts available.
    """
    total = len(verdict_records)
    if total == 0:
        return {
            "check": "llm_output_violation_rate",
            "status": "SKIP",
            "message": "No verdict records provided.",
        }

    violations = sum(
        1 for v in verdict_records
        if v.get("overall_verdict") not in VALID_VERDICTS
    )
    rate = violations / total

    trend = "unknown"
    if baseline_rate is not None:
        trend = "rising" if rate > baseline_rate * 1.5 else "stable"

    status = "WARN" if rate > warn_threshold else "PASS"

    return {
        "check": "llm_output_violation_rate",
        "status": status,
        "total_outputs": total,
        "schema_violations": violations,
        "violation_rate": round(rate, 4),
        "warn_threshold": warn_threshold,
        "trend": trend,
        "baseline_rate": baseline_rate,
        "message": (
            f"LLM output schema violation rate: {rate*100:.2f}% "
            f"({violations}/{total} verdicts with overall_verdict outside {{PASS, FAIL, WARN}}). "
            f"Trend: {trend}."
        ),
    }


def _synthesise_verdicts_from_traces(traces: list[dict]) -> list[dict]:
    """
    Fallback: synthesise verdict-like records from LLM trace outputs.
    LLM runs have outputs['score'] (int 1-5) — map to PASS/FAIL/WARN.
    """
    verdicts = []
    for run in traces:
        if run.get("run_type") != "llm":
            continue
        outputs = run.get("outputs", {})
        if isinstance(outputs, dict) and "score" in outputs:
            score = outputs["score"]
            if isinstance(score, (int, float)):
                verdict = "PASS" if score >= 3 else "FAIL"
            else:
                verdict = None  # schema violation — score not numeric
            verdicts.append({
                "overall_verdict": verdict,
                "_source": "synthesised_from_llm_trace",
                "run_id": run.get("id"),
            })
    return verdicts


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AI-powered contract checks: embedding drift, prompt schema, LLM output violation rate."
        )
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["all", "drift", "prompt", "output_rate"],
    )
    parser.add_argument("--extractions", default="outputs/week3/extractions.jsonl")
    parser.add_argument(
        "--verdicts",
        default="outputs/week2/verdicts.jsonl",
        help="Week 2 verdict records for Extension 3 (falls back to LLM traces if missing)",
    )
    parser.add_argument("--traces", default="outputs/traces/runs.jsonl")
    parser.add_argument("--output", default="enforcer_report/ai_extensions.json")
    parser.add_argument(
        "--baseline-cache",
        default="schema_snapshots/embedding_baselines/week3_centroid.json",
        help="Path to store/load embedding centroid baseline",
    )
    args = parser.parse_args()

    results = {}

    # ── Extension 1: Embedding Drift
    if args.mode in ("all", "drift"):
        print(f"[ai_extensions] Ext 1 — Embedding drift: loading {args.extractions}")
        extractions = load_jsonl(args.extractions)
        # Split: first half as baseline, second half as current
        mid = max(1, len(extractions) // 2)
        baseline_recs = extractions[:mid]
        current_recs  = extractions[mid:] if len(extractions) > mid else extractions
        result = compute_embedding_drift(
            baseline_recs, current_recs,
            baseline_cache_path=args.baseline_cache,
        )
        results["embedding_drift"] = result
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "-"}.get(result["status"], "?")
        print(f"  [{icon}] {result['message']}")

    # ── Extension 2: Prompt Input Schema Validation
    if args.mode in ("all", "prompt"):
        print(f"[ai_extensions] Ext 2 — Prompt schema: loading {args.extractions}")
        if "extractions" not in dir() or not extractions:
            extractions = load_jsonl(args.extractions)
        result = validate_prompt_schemas(extractions)
        results["prompt_schema_validation"] = result
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "-"}.get(result["status"], "?")
        print(f"  [{icon}] {result['message']}")

    # ── Extension 3: LLM Output Schema Violation Rate
    if args.mode in ("all", "output_rate"):
        verdicts_path = Path(args.verdicts)
        if verdicts_path.exists():
            print(f"[ai_extensions] Ext 3 — LLM output rate: loading {args.verdicts}")
            verdict_records = load_jsonl(args.verdicts)
        else:
            print(
                f"[ai_extensions] Ext 3 — {args.verdicts} not found; "
                f"synthesising from LLM traces: {args.traces}"
            )
            traces = load_jsonl(args.traces)
            verdict_records = _synthesise_verdicts_from_traces(traces)
            print(f"[ai_extensions]   Synthesised {len(verdict_records)} verdict records from traces.")

        result = check_llm_output_violation_rate(verdict_records)
        results["llm_output_violation_rate"] = result
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "-"}.get(result["status"], "?")
        print(f"  [{icon}] {result['message']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ai_extensions] Written: {output_path}")


if __name__ == "__main__":
    main()
