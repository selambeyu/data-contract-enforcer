"""
ContractGenerator — Stage 1-4 implementation.

Usage:
    python contracts/generator.py \
        --source outputs/week3/extractions.jsonl \
        --contract-id week3-document-refinery-extractions \
        --lineage outputs/week4/lineage_snapshots.jsonl \
        --output generated_contracts/

Stages:
    1. Load and profile the data
    2. Structural profiling per column (flatten nested arrays first)
    3. Translate column profiles to Bitol YAML clauses
    4. Inject lineage context and write YAML
"""

import argparse
import hashlib
import json
import math
import os
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

# Load .env from project root (src/contracts/ → project root)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ── LLM Annotator (Step 4 spec requirement) ───────────────────────────────────

class LLMAnnotator:
    """
    Calls an LLM to generate plain-English descriptions and business rules
    for contract columns whose meaning is ambiguous from name + sample values.

    Priority (from .env): OpenAI → OpenRouter → Ollama → None (skip annotation).
    """

    _SYSTEM = (
        "You are a senior data engineer annotating a data contract. "
        "Given a column name, its table context, sample values, and adjacent columns, "
        "return ONLY a JSON object with these exact keys:\n"
        '  "description": one plain-English sentence describing what this field contains,\n'
        '  "business_rule": one sentence stating the validation rule in plain English,\n'
        '  "cross_column_note": one sentence about any dependency on adjacent columns, '
        'or null if none.\n'
        "No extra text, no markdown fences, no preamble. Pure JSON only."
    )

    _OPENAI_MODEL        = "gpt-4o-mini"
    _OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    _OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"
    _OLLAMA_DEFAULT_HOST  = "http://localhost:11434"
    _OLLAMA_DEFAULT_MODEL = "llama3.2:3b"

    def __init__(self) -> None:
        self._backend: str | None = None
        self._openai_key       = os.environ.get("OPENAI_API_KEY", "").strip()
        self._openrouter_key   = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self._openrouter_model = os.environ.get("OPENROUTER_MODEL", self._OPENROUTER_DEFAULT_MODEL)
        self._ollama_host      = os.environ.get("OLLAMA_HOST", self._OLLAMA_DEFAULT_HOST).rstrip("/")
        self._ollama_model     = os.environ.get("OLLAMA_MODEL", self._OLLAMA_DEFAULT_MODEL)

        if self._openai_key:
            self._backend = "openai"
        elif self._openrouter_key:
            self._backend = "openrouter"
        elif self._ollama_reachable():
            self._backend = "ollama"

        if self._backend:
            print(f"[generator] LLM annotation backend: {self._backend}")
        else:
            print("[generator] No LLM available — skipping column annotation (set OPENAI_API_KEY, OPENROUTER_API_KEY, or start Ollama)")

    def _ollama_reachable(self) -> bool:
        try:
            req = urllib.request.Request(f"{self._ollama_host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3):
                return True
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self._backend is not None

    def annotate(self, col_name: str, table_name: str, sample_values: list, adjacent_cols: list) -> dict | None:
        """
        Returns dict with keys: description, business_rule, cross_column_note
        or None if LLM unavailable or call fails.
        """
        prompt = (
            f"Column name: {col_name}\n"
            f"Table/contract: {table_name}\n"
            f"Sample values (up to 5): {sample_values[:5]}\n"
            f"Adjacent columns: {adjacent_cols[:8]}\n\n"
            "Return a JSON object with keys: description, business_rule, cross_column_note."
        )
        raw = self._call(prompt)
        if not raw:
            return None
        try:
            # Strip markdown fences if model wraps in ```json
            clean = raw.strip().strip("`")
            if clean.startswith("json"):
                clean = clean[4:].strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            return None

    def _call(self, prompt: str) -> str | None:
        if self._backend == "openai":
            return self._openai(prompt)
        if self._backend == "openrouter":
            return self._openrouter(prompt)
        if self._backend == "ollama":
            return self._ollama(prompt)
        return None

    def _openai(self, prompt: str) -> str | None:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._openai_key)
            resp = client.chat.completions.create(
                model=self._OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=200,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[generator] LLM OpenAI error: {exc}")
            return None

    def _openrouter(self, prompt: str) -> str | None:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._openrouter_key, base_url=self._OPENROUTER_BASE_URL)
            resp = client.chat.completions.create(
                model=self._openrouter_model,
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=200,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[generator] LLM OpenRouter error: {exc}")
            return None

    def _ollama(self, prompt: str) -> str | None:
        try:
            payload = json.dumps({
                "model": self._ollama_model,
                "prompt": self._SYSTEM + "\n\n" + prompt,
                "stream": False,
                "options": {"num_predict": 200, "temperature": 0.1},
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
            print(f"[generator] LLM Ollama error: {exc}")
            return None


def annotate_schema_with_llm(
    schema_block: dict,
    column_profiles: dict,
    contract_id: str,
    annotator: LLMAnnotator,
    skip_obvious: bool = True,
) -> dict:
    """
    Step 4 (spec): for any column whose business meaning is ambiguous from
    name + sample values alone, invoke the LLM and append llm_annotations
    block to that clause.

    skip_obvious=True skips columns whose meaning is already clear:
      _id, _at, _hash columns, and columns with descriptions already set.
    """
    if not annotator.available:
        return schema_block

    all_cols = list(schema_block.keys())

    for col, clause in schema_block.items():
        # Skip columns with obvious meaning
        if skip_obvious:
            if clause.get("description") and "Auto-generated" not in clause.get("description", ""):
                continue  # already has a non-heuristic description
            if any(col.endswith(sfx) for sfx in ("_id", "_at", "_hash")):
                continue
            if clause.get("type") == "boolean":
                continue

        # Determine adjacent columns for context
        idx = all_cols.index(col)
        adjacent = all_cols[max(0, idx - 3): idx] + all_cols[idx + 1: idx + 4]

        profile   = column_profiles.get(col, {})
        samples   = profile.get("sample_values", [])

        print(f"[generator]   Annotating '{col}' via LLM ...")
        annotation = annotator.annotate(
            col_name=col,
            table_name=contract_id,
            sample_values=samples,
            adjacent_cols=adjacent,
        )

        if annotation:
            clause["llm_annotations"] = {
                "description":      annotation.get("description", ""),
                "business_rule":    annotation.get("business_rule", ""),
                "cross_column_note": annotation.get("cross_column_note"),
                "annotated_by":     "llm",
            }
            # Promote description to top-level if not already set
            if not clause.get("description"):
                clause["description"] = annotation.get("description", "")
        else:
            print(f"[generator]   WARNING: LLM annotation failed for '{col}', skipping.")

    return schema_block


# ── Stage 1: Load ────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _pick_explode_key(record: dict) -> str | None:
    """
    Auto-detect the best array field to explode for profiling.
    Picks the list field whose items are dicts and that has the most total items
    across all records (most schema-rich). Skips _migration_meta.
    Priority: extracted_facts > nodes > edges > events > items > first list found.
    """
    PREFERRED = ["extracted_facts", "nodes", "edges", "events", "items", "records"]
    for key in PREFERRED:
        if key in record and isinstance(record[key], list):
            return key
    # Fallback: first list-of-dicts field
    for k, v in record.items():
        if k.startswith("_"):
            continue
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return k
    return None


def flatten_for_profile(records: list[dict]) -> pd.DataFrame:
    """
    Flatten nested JSONL to a flat DataFrame for profiling.
    Auto-detects the primary array field (extracted_facts, nodes, edges, etc.)
    and explodes it to one row per item.
    Nested dicts within each item are prefixed with the array field name.
    Strips _migration_meta fields (internal scaffolding).
    """
    if not records:
        return pd.DataFrame()

    _PREFIX_MAP = {
        "extracted_facts": "fact_",
        "nodes": "node_",
        "edges": "edge_",
        "events": "event_",
        "items": "item_",
        "records": "record_",
    }
    explode_key = _pick_explode_key(records[0])
    prefix = _PREFIX_MAP.get(explode_key, explode_key.rstrip("s") + "_") if explode_key else ""

    rows = []
    for r in records:
        base = {
            k: v for k, v in r.items()
            if not isinstance(v, (list, dict)) and not k.startswith("_")
        }
        items = r.get(explode_key, []) if explode_key else []
        if items:
            for item in items:
                if isinstance(item, dict):
                    item_flat = {
                        f"{prefix}{k}": v
                        for k, v in item.items()
                        if not isinstance(v, (list, dict))
                    }
                    rows.append({**base, **item_flat})
                else:
                    rows.append({**base, f"{prefix}value": item})
        else:
            rows.append(base)

    df = pd.DataFrame(rows)
    df = df.dropna(axis=1, how="all")
    return df


# ── Stage 2: Column profiling ─────────────────────────────────────────────────

def profile_column(series: pd.Series, col_name: str) -> dict:
    result: dict[str, Any] = {
        "name": col_name,
        "dtype": str(series.dtype),
        "null_fraction": float(series.isna().mean()),
        "cardinality_estimate": int(series.nunique()),
        "sample_values": [str(v) for v in series.dropna().unique()[:5]],
    }
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            result.update({
                "mean": float(non_null.mean()),
                "p25": float(non_null.quantile(0.25)),
                "p50": float(non_null.quantile(0.50)),
                "p75": float(non_null.quantile(0.75)),
                "p95": float(non_null.quantile(0.95)),
                "p99": float(non_null.quantile(0.99)),
                "stddev": float(non_null.std()) if len(non_null) > 1 else 0.0,
                "min": float(non_null.min()),
                "max": float(non_null.max()),
            })
    return result


def profile_all_columns(df: pd.DataFrame) -> dict[str, dict]:
    return {col: profile_column(df[col], col) for col in df.columns}


# ── Stage 3: Profile → Bitol YAML clause ─────────────────────────────────────

def infer_type(dtype_str: str) -> str:
    mapping = {
        "float64": "number",
        "float32": "number",
        "int64": "integer",
        "int32": "integer",
        "int16": "integer",
        "bool": "boolean",
        "object": "string",
    }
    return mapping.get(dtype_str, "string")


def _distribution_warnings(profile: dict, col_name: str) -> list[dict]:
    """
    Detect suspicious numeric distributions and return a list of warning
    annotations that are embedded directly in the contract clause.

    Each warning is a dict with keys: code, message, severity.
    Downstream tooling (ValidationRunner, ReportGenerator) can react to these
    automatically without re-reading the source data.

    Checks performed:
      1. near_constant_high  — mean > 0.99 on a 0-1 field: all scores are
         near-maximum, suggesting the model is over-confident or the field
         was incorrectly normalised from a higher range.
      2. near_constant_low   — mean < 0.01 on a 0-1 field: all scores are
         near-zero, suggesting a scale inversion or uninitialized field.
      3. low_variance        — stddev < 0.001 on a numeric field with
         cardinality > 5: values are effectively constant despite appearing
         to vary, suggesting a bug in the producer.
      4. extreme_skew        — p99 > 10 × p50 for non-constant fields:
         heavily right-skewed distribution, often caused by log-scale
         confusion or outlier injection.
      5. unexpected_negatives — min < 0 on a field that should be non-negative
         (confidence, processing_time, counts).
    """
    warnings = []
    dtype = profile.get("dtype", "")
    if not any(t in dtype for t in ("float", "int")):
        return warnings

    mean   = profile.get("mean")
    stddev = profile.get("stddev")
    p50    = profile.get("p50")
    p99    = profile.get("p99")
    mn     = profile.get("min")
    mx     = profile.get("max")

    if mean is None:
        return warnings

    # 1. near_constant_high — mean > 0.99 on bounded 0–1 fields
    if mx is not None and float(mx) <= 1.0 and mean > 0.99:
        warnings.append({
            "code":     "near_constant_high",
            "severity": "WARN",
            "message":  (
                f"Column '{col_name}': mean={mean:.4f} is suspiciously close to maximum "
                f"(max={mx}). All values are near the upper bound. Possible causes: "
                f"model over-confidence, scale already in 0–100 range normalised back to 0–1, "
                f"or field incorrectly populated with a constant."
            ),
        })

    # 2. near_constant_low — mean < 0.01 on bounded 0–1 fields
    if mx is not None and float(mx) <= 1.0 and mean < 0.01:
        warnings.append({
            "code":     "near_constant_low",
            "severity": "WARN",
            "message":  (
                f"Column '{col_name}': mean={mean:.4f} is suspiciously close to zero "
                f"(max={mx}). Possible causes: scale inversion (100 → 0.01 after ÷100), "
                f"uninitialised field, or all extractions failed silently."
            ),
        })

    # 3. low_variance — stddev nearly zero despite multiple distinct values
    cardinality = profile.get("cardinality_estimate", 0)
    if stddev is not None and stddev < 0.001 and cardinality > 5:
        warnings.append({
            "code":     "low_variance",
            "severity": "WARN",
            "message":  (
                f"Column '{col_name}': stddev={stddev:.6f} with cardinality={cardinality}. "
                f"Values appear effectively constant despite {cardinality} distinct entries. "
                f"Possible bug in producer rounding or clamping logic."
            ),
        })

    # 4. extreme_skew — p99 > 10× p50 (non-zero median)
    if p50 is not None and p99 is not None and p50 > 0 and p99 > 10 * p50:
        warnings.append({
            "code":     "extreme_skew",
            "severity": "WARN",
            "message":  (
                f"Column '{col_name}': p99={p99:.2f} is {p99/p50:.0f}× the median "
                f"p50={p50:.2f}. Heavily right-skewed. Possible causes: outlier injection, "
                f"log-scale confusion, or the field mixes two distinct populations."
            ),
        })

    # 5. unexpected_negatives — min < 0 on fields that should be non-negative
    _NON_NEGATIVE_HINTS = ("confidence", "time_ms", "count", "rate", "score", "tokens", "cost")
    if mn is not None and float(mn) < 0:
        if any(hint in col_name.lower() for hint in _NON_NEGATIVE_HINTS):
            warnings.append({
                "code":     "unexpected_negatives",
                "severity": "HIGH",
                "message":  (
                    f"Column '{col_name}': min={mn:.4f} is negative. "
                    f"This field is expected to be non-negative. "
                    f"Possible causes: signed-integer overflow, missing abs(), "
                    f"or delta values being written instead of absolute values."
                ),
            })

    return warnings


def column_to_clause(profile: dict) -> dict:
    clause: dict[str, Any] = {
        "type": infer_type(profile["dtype"]),
        "required": profile["null_fraction"] == 0.0,
    }

    # Rule 1: confidence columns must be 0.0–1.0 float
    if "confidence" in profile["name"] and clause["type"] == "number":
        clause["minimum"] = 0.0
        clause["maximum"] = 1.0
        clause["description"] = (
            "Confidence score. Must remain 0.0–1.0 float. "
            "BREAKING CHANGE if changed to 0–100 integer percentage."
        )

    # Rule 2: UUID format for _id columns
    if profile["name"].endswith("_id"):
        clause["format"] = "uuid"
        clause["pattern"] = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

    # Rule 3: date-time format for _at columns
    if profile["name"].endswith("_at"):
        clause["format"] = "date-time"
        clause["description"] = clause.get("description", "") or "ISO 8601 datetime string."

    # Rule 4: enum for low-cardinality string columns
    # Only add enum if sample_values covers the full cardinality
    if (
        profile["cardinality_estimate"] <= 10
        and clause["type"] == "string"
        and not profile["name"].endswith("_id")
        and not profile["name"].endswith("_at")
        and len(profile["sample_values"]) >= profile["cardinality_estimate"]
    ):
        clause["enum"] = sorted(profile["sample_values"])

    # Rule 5: numeric range annotation (non-confidence columns)
    if clause["type"] in ("number", "integer") and "min" in profile and "confidence" not in profile["name"]:
        if profile["min"] >= 0:
            clause["minimum"] = 0

    # Rule 6: embed numeric baseline statistics into the clause
    # These are persisted here so downstream tooling (ValidationRunner,
    # SchemaEvolutionAnalyzer) can read them directly from the contract
    # without accessing the original source data.
    if profile.get("mean") is not None:
        clause["baseline_statistics"] = {
            "mean":   round(profile["mean"],   6),
            "stddev": round(profile.get("stddev", 0.0), 6),
            "min":    round(profile.get("min",  0.0), 6),
            "max":    round(profile.get("max",  0.0), 6),
            "p25":    round(profile.get("p25",  0.0), 6),
            "p50":    round(profile.get("p50",  0.0), 6),
            "p75":    round(profile.get("p75",  0.0), 6),
            "p95":    round(profile.get("p95",  0.0), 6),
            "p99":    round(profile.get("p99",  0.0), 6),
        }

    # Rule 7: distribution warning annotations for suspicious distributions
    dist_warnings = _distribution_warnings(profile, profile["name"])
    if dist_warnings:
        clause["distribution_warnings"] = dist_warnings

    return clause


def build_schema_block(column_profiles: dict[str, dict]) -> dict:
    return {col: column_to_clause(profile) for col, profile in column_profiles.items()}


# ── Stage 4: Lineage injection (registry-first) ───────────────────────────────

def load_registry(registry_path: str | Path) -> list[dict]:
    """Load subscriptions.yaml and return list of subscription dicts."""
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return []
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    return data.get("subscriptions", [])


def inject_lineage(
    contract: dict,
    lineage_path: str | Path,
    source_path: str,
    registry_path: str | Path = "contract_registry/subscriptions.yaml",
) -> dict:
    """
    Stage 4: Inject lineage context into contract.

    PRIMARY source: contract_registry/subscriptions.yaml
      → consumers who registered a subscription for this contract_id

    ENRICHMENT: lineage_snapshots.jsonl
      → used to annotate downstream nodes with labels from the graph
      → does NOT determine who the consumers are (registry does that)
    """
    contract_id = contract.get("id", "")

    # ── PRIMARY: read from registry ──────────────────────────────────────────
    try:
        subscriptions = load_registry(registry_path)
        registry_subscribers = [
            s for s in subscriptions if s.get("contract_id") == contract_id
        ]
    except Exception as exc:
        registry_subscribers = []
        print(f"[generator] WARNING: Could not load registry: {exc}")

    # ── ENRICHMENT: lineage graph for node labels ─────────────────────────────
    node_labels: dict[str, str] = {}
    try:
        with open(lineage_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            snapshot = json.loads(lines[-1])
            node_labels = {
                n["node_id"]: n.get("label", n["node_id"])
                for n in snapshot.get("nodes", [])
            }
    except Exception:
        pass  # enrichment is optional

    # ── Build downstream list from registry subscribers ───────────────────────
    downstream = []
    for sub in registry_subscribers:
        subscriber_id = sub.get("subscriber_id", "")
        # Try to enrich with a label from the lineage graph
        label = node_labels.get(subscriber_id, subscriber_id)
        downstream.append({
            "id": subscriber_id,
            "label": label,
            "fields_consumed": sub.get("fields_consumed", []),
            "breaking_fields": sub.get("breaking_fields", []),
            "tier": sub.get("tier", 1),
        })

    contract["lineage"] = {
        "upstream": [],
        "downstream": downstream,
    }

    return contract


# ── Build full contract dict ──────────────────────────────────────────────────

def build_contract(
    column_profiles: dict[str, dict],
    contract_id: str,
    source_path: str,
    records: list[dict],
) -> dict:
    schema_block = build_schema_block(column_profiles)

    # Build Soda checks from schema
    soda_checks = []
    for col, clause in schema_block.items():
        if clause.get("required"):
            soda_checks.append(f"missing_count({col}) = 0")
        if clause.get("format") == "uuid" and clause.get("required"):
            soda_checks.append(f"duplicate_count({col}) = 0")

    # Snapshot hash of the source data for reproducibility
    snapshot_hash = hashlib.sha256(
        json.dumps(records[:10], sort_keys=True).encode()
    ).hexdigest()[:16]

    contract = {
        "kind": "DataContract",
        "apiVersion": "v3.0.0",
        "id": contract_id,
        "info": {
            "title": _contract_title(contract_id),
            "version": "1.0.0",
            "owner": "week7-enforcer",
            "description": (
                f"Auto-generated contract for {source_path}. "
                f"Source snapshot: {snapshot_hash}. "
                f"Generated at: {datetime.now(timezone.utc).isoformat()}."
            ),
        },
        "servers": {
            "local": {
                "type": "local",
                "path": str(source_path),
                "format": "jsonl",
            }
        },
        "terms": {
            "usage": "Internal inter-system data contract. Do not publish.",
            "limitations": (
                "All confidence fields must remain in 0.0–1.0 float range. "
                "All _id fields must be valid UUID v4."
            ),
        },
        "schema": schema_block,
        "quality": {
            "type": "SodaChecks",
            "specification": {
                f"checks for {Path(source_path).stem}": soda_checks,
            },
        },
    }
    return contract


def _contract_title(contract_id: str) -> str:
    return contract_id.replace("-", " ").replace("_", " ").title()


# ── Baseline statistics artifact ─────────────────────────────────────────────

def write_numeric_baselines(
    column_profiles: dict[str, dict],
    contract_id: str,
    baselines_path: str | Path = "schema_snapshots/baselines.json",
) -> Path:
    """
    Persist mean and stddev for every numeric column to a dedicated
    schema_snapshots/baselines.json artifact.

    The ValidationRunner reads this file to compute z-score drift checks.
    The SchemaEvolutionAnalyzer reads it to detect scale mutations between
    generator runs without needing the original source data.

    The file is a flat dict keyed by "{contract_id}.{column_name}" so that
    multiple contracts can share a single baselines.json without collision:

    {
      "week3-document-refinery-extractions.fact_confidence": {
        "mean": 0.8983, "stddev": 0.0421, "min": 0.65, "max": 0.9,
        "p25": 0.88, "p50": 0.9, "p75": 0.9, "p95": 0.9, "p99": 0.9,
        "captured_at": "2026-04-04T12:00:00+00:00",
        "contract_id": "week3-document-refinery-extractions",
        "column": "fact_confidence",
        "distribution_warnings": []
      },
      ...
    }

    Calling this function MERGES the new entries into the existing file —
    previous contract baselines are preserved unless overwritten.
    """
    baselines_path = Path(baselines_path)
    baselines_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing baselines (if any)
    existing: dict = {}
    if baselines_path.exists():
        try:
            with open(baselines_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    now_iso = datetime.now(timezone.utc).isoformat()
    updated = 0

    for col, profile in column_profiles.items():
        if profile.get("mean") is None:
            continue  # skip non-numeric columns

        key = f"{contract_id}.{col}"
        dist_warnings = _distribution_warnings(profile, col)

        existing[key] = {
            "contract_id":  contract_id,
            "column":       col,
            "mean":         round(profile["mean"],            6),
            "stddev":       round(profile.get("stddev", 0.0), 6),
            "min":          round(profile.get("min",   0.0),  6),
            "max":          round(profile.get("max",   0.0),  6),
            "p25":          round(profile.get("p25",   0.0),  6),
            "p50":          round(profile.get("p50",   0.0),  6),
            "p75":          round(profile.get("p75",   0.0),  6),
            "p95":          round(profile.get("p95",   0.0),  6),
            "p99":          round(profile.get("p99",   0.0),  6),
            "captured_at":  now_iso,
            "distribution_warnings": dist_warnings,
        }
        updated += 1

    with open(baselines_path, "w") as f:
        json.dump(existing, f, indent=2)

    return baselines_path


# ── Write YAML ────────────────────────────────────────────────────────────────

def write_yaml(contract: dict, output_dir: str | Path, contract_id: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = contract_id.replace(" ", "_")
    output_path = output_dir / f"{safe_name}.yaml"
    with open(output_path, "w") as f:
        yaml.dump(contract, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return output_path


# ── Write timestamped schema snapshot ────────────────────────────────────────

def write_schema_snapshot(
    schema_block: dict,
    contract_id: str,
    snapshots_root: str | Path = "schema_snapshots",
) -> Path:
    """
    Write a timestamped snapshot of the inferred schema to:
      schema_snapshots/{contract_id}/{timestamp}.yaml
    Called on every ContractGenerator run so SchemaEvolutionAnalyzer can diff
    consecutive snapshots to detect changes over time.
    """
    snap_dir = Path(snapshots_root) / contract_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_path = snap_dir / f"{ts}.yaml"
    snapshot = {
        "contract_id": contract_id,
        "snapshot_version": ts,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "schema": schema_block,
    }
    with open(snap_path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return snap_path


# ── Write dbt schema.yml ──────────────────────────────────────────────────────

def write_dbt_schema(
    schema_block: dict,
    contract_id: str,
    output_dir: str | Path,
) -> Path:
    """
    Generate a dbt-compatible schema.yml alongside the Bitol contract.
    Maps: required → not_null, enum → accepted_values, uuid format → unique.
    Place in generated_contracts/{contract_id}_dbt.yml
    """
    columns = []
    for col, clause in schema_block.items():
        col_entry: dict = {"name": col}
        desc = clause.get("description", "")
        if desc:
            col_entry["description"] = desc
        col_tests: list = []
        if clause.get("required"):
            col_tests.append("not_null")
        if clause.get("enum"):
            col_tests.append({"accepted_values": {"values": clause["enum"]}})
        if clause.get("format") == "uuid":
            col_tests.append("unique")
        if col_tests:
            col_entry["tests"] = col_tests
        columns.append(col_entry)

    dbt_doc = {
        "version": 2,
        "models": [
            {
                "name": contract_id,
                "description": f"Auto-generated dbt schema from Bitol contract {contract_id}",
                "columns": columns,
            }
        ],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dbt_path = output_dir / f"{contract_id}_dbt.yml"
    with open(dbt_path, "w") as f:
        yaml.dump(dbt_doc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return dbt_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-generate a Bitol data contract from JSONL data.")
    parser.add_argument("--source", required=True, help="Path to source JSONL file")
    parser.add_argument("--contract-id", required=True, help="Unique contract identifier")
    parser.add_argument("--lineage", required=True, help="Path to lineage_snapshots.jsonl")
    parser.add_argument("--output", required=True, help="Directory to write generated contract YAML")
    parser.add_argument(
        "--registry",
        default="contract_registry/subscriptions.yaml",
        help="Path to contract registry subscriptions.yaml",
    )
    parser.add_argument(
        "--snapshots",
        default="schema_snapshots",
        help="Root dir for timestamped schema snapshots (default: schema_snapshots/)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM annotation step even if a backend is available",
    )
    parser.add_argument(
        "--baselines",
        default="schema_snapshots/baselines.json",
        help="Path to persist numeric baseline statistics (default: schema_snapshots/baselines.json)",
    )
    args = parser.parse_args()

    print(f"[generator] Loading {args.source} ...")
    records = load_jsonl(args.source)
    print(f"[generator] Loaded {len(records)} records.")

    # ── Stage 1: Flatten
    print("[generator] Stage 1: Flattening for profile ...")
    df = flatten_for_profile(records)
    print(f"[generator] Flattened DataFrame shape: {df.shape}")
    print(f"[generator] Columns: {list(df.columns)}")

    # ── Warn on mixed types
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(20)
            numeric_count = sum(
                1 for v in sample if str(v).replace(".", "", 1).lstrip("-").isdigit()
            )
            if 0 < numeric_count < len(sample):
                print(
                    f"[generator] WARNING: column '{col}' has mixed types "
                    f"(dtype=object but some values appear numeric). Document as contract violation."
                )

    # ── Stage 2: Profile
    print("[generator] Stage 2: Profiling columns ...")
    column_profiles = profile_all_columns(df)
    for col, profile in column_profiles.items():
        null_pct = profile["null_fraction"] * 100
        print(
            f"  {col}: type={profile['dtype']}, nulls={null_pct:.1f}%, "
            f"cardinality={profile['cardinality_estimate']}"
        )

    # ── Stage 2.5: Distribution warning scan
    all_warnings = []
    for col, profile in column_profiles.items():
        warnings = _distribution_warnings(profile, col)
        for w in warnings:
            print(f"[generator] DIST WARNING [{w['severity']}] {w['code']}: {w['message'][:100]}")
            all_warnings.append(w)
    if not all_warnings:
        print("[generator] Stage 2.5: No suspicious distributions detected.")

    # ── Stage 3: Build contract
    print("[generator] Stage 3: Building Bitol YAML clauses ...")
    contract = build_contract(column_profiles, args.contract_id, args.source, records)

    # ── Stage 4: Inject lineage (registry-first)
    print(f"[generator] Stage 4: Injecting lineage from registry {args.registry} ...")
    contract = inject_lineage(contract, args.lineage, args.source, args.registry)
    downstream_count = len(contract.get("lineage", {}).get("downstream", []))
    print(f"[generator] Found {downstream_count} downstream subscribers from registry.")

    # ── Stage 4.5: LLM annotation (Step 4 spec — ambiguous columns)
    if not args.no_llm:
        print("[generator] Stage 4.5: LLM annotation of ambiguous columns ...")
        annotator = LLMAnnotator()
        contract["schema"] = annotate_schema_with_llm(
            contract["schema"],
            column_profiles,
            args.contract_id,
            annotator,
        )
    else:
        print("[generator] Stage 4.5: LLM annotation skipped (--no-llm)")

    # ── Write contract YAML
    output_path = write_yaml(contract, args.output, args.contract_id)
    clause_count = len(contract["schema"])
    print(f"[generator] Contract YAML: {clause_count} clauses → {output_path}")

    # ── Write dedicated numeric baselines artifact
    baselines_path = write_numeric_baselines(
        column_profiles, args.contract_id, args.baselines
    )
    numeric_cols = sum(1 for p in column_profiles.values() if p.get("mean") is not None)
    print(f"[generator] Numeric baselines → {baselines_path} ({numeric_cols} columns persisted)")

    # ── Write dbt schema.yml counterpart
    dbt_path = write_dbt_schema(contract["schema"], args.contract_id, args.output)
    print(f"[generator] dbt schema.yml  → {dbt_path}")

    # ── Write timestamped schema snapshot
    snap_path = write_schema_snapshot(contract["schema"], args.contract_id, args.snapshots)
    print(f"[generator] Schema snapshot → {snap_path}")

    if clause_count < 6:
        print(
            f"[generator] WARNING: only {clause_count} clauses — "
            f"minimum is 6 (week5) or 8 (week3). Check source data."
        )


if __name__ == "__main__":
    main()
