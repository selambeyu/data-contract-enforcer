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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


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

    return clause


def build_schema_block(column_profiles: dict[str, dict]) -> dict:
    return {col: column_to_clause(profile) for col, profile in column_profiles.items()}


# ── Stage 4: Lineage injection ────────────────────────────────────────────────

def inject_lineage(contract: dict, lineage_path: str | Path, source_path: str) -> dict:
    try:
        with open(lineage_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            contract["lineage"] = {"upstream": [], "downstream": []}
            return contract

        snapshot = json.loads(lines[-1])  # latest snapshot
        edges = snapshot.get("edges", [])
        nodes = snapshot.get("nodes", [])

        # Build node_id -> label map
        node_labels = {n["node_id"]: n.get("label", n["node_id"]) for n in nodes}

        # Source identifier keywords derived from the source path
        source_keywords = Path(source_path).stem.replace("_", " ").lower().split()
        source_keywords += ["week3", "extraction", "refinery"]

        # Find edges whose source matches the source system's output
        consumers = []
        for e in edges:
            src_lower = str(e.get("source", "")).lower()
            if any(kw in src_lower for kw in source_keywords):
                target_id = e["target"]
                consumers.append({
                    "id": target_id,
                    "label": node_labels.get(target_id, target_id),
                    "fields_consumed": ["doc_id", "extracted_facts", "extraction_model"],
                })

        contract["lineage"] = {
            "upstream": [],
            "downstream": consumers[:10],  # cap at 10 for readability
        }
    except Exception as exc:
        contract["lineage"] = {
            "upstream": [],
            "downstream": [],
            "_error": f"Could not load lineage: {exc}",
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


# ── Write YAML ────────────────────────────────────────────────────────────────

def write_yaml(contract: dict, output_dir: str | Path, contract_id: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # contract_id may contain hyphens; use as filename stem
    safe_name = contract_id.replace(" ", "_")
    output_path = output_dir / f"{safe_name}.yaml"
    with open(output_path, "w") as f:
        yaml.dump(contract, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-generate a Bitol data contract from JSONL data.")
    parser.add_argument("--source", required=True, help="Path to source JSONL file")
    parser.add_argument("--contract-id", required=True, help="Unique contract identifier")
    parser.add_argument("--lineage", required=True, help="Path to lineage_snapshots.jsonl")
    parser.add_argument("--output", required=True, help="Directory to write generated contract YAML")
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
            # Check if there are actually numeric strings mixed in
            sample = df[col].dropna().head(20)
            numeric_count = sum(1 for v in sample if str(v).replace(".", "", 1).lstrip("-").isdigit())
            if numeric_count > 0 and numeric_count < len(sample):
                print(f"[generator] WARNING: column '{col}' has mixed types (dtype=object but some values appear numeric). Document as contract violation.")

    # ── Stage 2: Profile
    print("[generator] Stage 2: Profiling columns ...")
    column_profiles = profile_all_columns(df)
    for col, profile in column_profiles.items():
        null_pct = profile["null_fraction"] * 100
        print(f"  {col}: type={profile['dtype']}, nulls={null_pct:.1f}%, cardinality={profile['cardinality_estimate']}")

    # ── Stage 3: Build contract
    print("[generator] Stage 3: Building Bitol YAML clauses ...")
    contract = build_contract(column_profiles, args.contract_id, args.source, records)

    # ── Stage 4: Inject lineage
    print(f"[generator] Stage 4: Injecting lineage from {args.lineage} ...")
    contract = inject_lineage(contract, args.lineage, args.source)
    downstream_count = len(contract.get("lineage", {}).get("downstream", []))
    print(f"[generator] Found {downstream_count} downstream consumers in lineage graph.")

    # ── Write
    output_path = write_yaml(contract, args.output, args.contract_id)
    clause_count = len(contract["schema"])
    print(f"[generator] Written {clause_count} schema clauses to {output_path}")

    if clause_count < 6:
        print(f"[generator] WARNING: only {clause_count} clauses — minimum is 6 (week5) or 8 (week3). Check source data.")


if __name__ == "__main__":
    main()
